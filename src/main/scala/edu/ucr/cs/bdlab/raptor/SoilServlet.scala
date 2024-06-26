package edu.ucr.cs.bdlab.raptor

import com.fasterxml.jackson.core.JsonFactory
import com.fasterxml.jackson.databind.ObjectMapper
import edu.ucr.cs.bdlab.beast.cg.Reprojector
import edu.ucr.cs.bdlab.beast.common.{BeastOptions, WebMethod}
import edu.ucr.cs.bdlab.beast.geolite.RasterMetadata
import edu.ucr.cs.bdlab.beast.indexing.RTreeFeatureReader
import edu.ucr.cs.bdlab.beast.io.{GeoJSONFeatureReader, SpatialFileRDD}
import edu.ucr.cs.bdlab.beast.util.AbstractWebHandler
import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapreduce.lib.input.FileSplit
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.geotools.referencing.operation.transform.{AffineTransform2D, ConcatenatedTransform}
import org.locationtech.jts.geom.{Envelope, Geometry, GeometryFactory}
import org.opengis.referencing.operation.MathTransform

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO
import javax.servlet.http.{HttpServletRequest, HttpServletResponse}
import scala.collection.JavaConverters.asScalaIteratorConverter

class SoilServlet extends AbstractWebHandler with Logging {
  import SoilServlet._

  /** Additional options passed on by the user to override existing options */
  var opts: BeastOptions = _

  /** The SparkSession that is used to process datasets */
  var sparkSession: SparkSession = _

  /** The path at which this server keeps all datasets */
  var dataPath: String = _

  override def setup(ss: SparkSession, opts: BeastOptions): Unit = {
    super.setup(ss, opts)
    this.opts = opts
    this.sparkSession = ss
    this.dataPath = opts.getString("datapath", "data")

    // Build indexes if not existent
    logInfo("Building raster indexes")
    val dataPath = new Path(new Path(this.dataPath), "POLARIS")
    val sc = ss.sparkContext
    val fs = dataPath.getFileSystem(sc.hadoopConfiguration)
    val directories = fs.globStatus(new Path(dataPath, "**/**"),
      (path: Path) => path.getName.matches("\\d+_\\d+_compressed"))
    for (dir <- directories) {
      val indexPath = new Path(dir.getPath, "_index.csv")
      if (!fs.exists(indexPath)) {
        logInfo(s"Building a raster index for '${dir.getPath}'")
        RasterFileRDD.buildIndex(sc, dir.getPath.toString, indexPath.toString)
      }
    }
  }

  @WebMethod(url = "/soil/singlepolygon.json", order = 1)
  def singlePolygon(path: String, request: HttpServletRequest, response: HttpServletResponse): Boolean = {
    val t1 = System.nanoTime()
    // sidebar select parameters
    var soilDepth = ""
    var layer = ""

    // try getting parameters from url
    try {
      soilDepth = request.getParameter("soildepth")
      layer = request.getParameter("layer")
    } catch {
      case e: NullPointerException => throw new RuntimeException("Couldn't find the required parameters: soildepth and layer")
    }

    // set content-type as application/json// set content-type as application/json
    response.setContentType("application/json")
    response.setStatus(HttpServletResponse.SC_OK)

    // load raster data based on selected soil depth and layer// load raster data based on selected soil depth and layer
    val matchingRasterDirs: Array[String] = rasterFiles
      .filter(rasterFile => SoilServlet.rangeOverlap(rasterFile._1, soilDepth))
      .map(rasterFile => s"data/POLARIS/$layer/${rasterFile._2}")
      .toArray

    val baos = new ByteArrayOutputStream
    val input = request.getInputStream
    IOUtils.copy(input, baos)
    input.close()
    baos.close()
    val geoJSONData: Array[Byte] = baos.toByteArray
    var geom: Geometry = null
    try {
      val jsonParser = new JsonFactory().createParser(new ByteArrayInputStream(geoJSONData))
      geom = GeoJSONFeatureReader.parseGeometry(jsonParser)
      geom.setSRID(4326)
      jsonParser.close()
    } catch {
      case e: Exception =>
        logError(s"Error parsing GeoJSON geometry ${new String(geoJSONData)}", e)
        throw new RuntimeException(s"Error parsing query geometry ${new String(geoJSONData)}", e)
    }

    // Now that we have a geometry object, call single machine raptor join
    val fileSystem = new Path(dataPath).getFileSystem(sparkSession.sparkContext.hadoopConfiguration)
    val matchingFiles = matchingRasterDirs.flatMap(matchingRasterDir =>
      RasterFileRDD.selectFiles(fileSystem, matchingRasterDir, geom))
    logDebug(s"Query matched ${matchingFiles.length} files")
    val singleMachineResults: Iterator[(Int, SingleMachineRaptorJoin.Statistics)] =
      SingleMachineRaptorJoin.zonalStatistics(matchingFiles, Array(geom), () => {(System.nanoTime() - t1) > ServerTimeout})
    val statistics: SingleMachineRaptorJoin.Statistics = if (singleMachineResults == null || singleMachineResults.isEmpty) {
      SingleMachineRaptorJoin.emptyStatistics
    } else {
      singleMachineResults.toArray.head._2
    }
    if (System.nanoTime() - t1 > ServerTimeout) {
      sendTimeout(response, System.nanoTime() - t1)
      return true
    }

    // Write result to json object
    val resWriter = response.getWriter
    val mapper = new ObjectMapper

    // Create query node
    val queryNode = mapper.createObjectNode
    queryNode.put("soildepth", soilDepth)
    queryNode.put("layer", layer)

    // create results node// create results node
    val resultsNode = mapper.createObjectNode
    if (singleMachineResults != null) {
      resultsNode.put("min", statistics.min)
      resultsNode.put("max", statistics.max)
      resultsNode.put("median", statistics.median)
      resultsNode.put("sum", statistics.sum)
      resultsNode.put("mode", statistics.mode)
      resultsNode.put("stddev", statistics.stdev)
      resultsNode.put("count", statistics.count)
      resultsNode.put("mean", statistics.mean)
      resultsNode.put("lowerquart", statistics.lowerQuart)
      resultsNode.put("upperquart", statistics.upperQuart)
    }

    // create root node
    // contains queryNode and resultsNode
    val rootNode = mapper.createObjectNode
    rootNode.set("query", queryNode)
    rootNode.set("results", resultsNode)

    // write values to response writer
    val jsonString = mapper.writer.writeValueAsString(rootNode)
    resWriter.print(jsonString)
    resWriter.flush()
    logInfo(s"Single polygon processing took ${(System.nanoTime() - t1)*1E-9} seconds")
    true
  }

  @WebMethod(url = "/soil/{datasetID}.json", order = 99)
  def queryVector(path: String, request: HttpServletRequest, response: HttpServletResponse, datasetID: String): Boolean = {
    // time at start of GET request
    val t1 = System.nanoTime

    response.setContentType("application/json")
    response.setStatus(HttpServletResponse.SC_OK)

    // Load the Farmland features
    val indexPath = new Path(VectorServlet.getIndexPathById(VectorServlet.VectorIndexFile, datasetID))
    val reader = new RTreeFeatureReader
    val fs = indexPath.getFileSystem(sparkSession.sparkContext.hadoopConfiguration)
    val fileLength = fs.getFileStatus(indexPath).getLen
    val inputFileSplit = new FileSplit(indexPath, 0, fileLength, null)
    val opts = new BeastOptions
    var soilDepth: String = null
    var layer: String = null
    var mbr: Envelope = null
    try {
      // get sidebar select parameters
      soilDepth = request.getParameter("soildepth")
      layer = request.getParameter("layer")
      // get extents parameters
      val minx = request.getParameter("minx").toDouble
      val miny = request.getParameter("miny").toDouble
      val maxx = request.getParameter("maxx").toDouble
      val maxy = request.getParameter("maxy").toDouble
      mbr = new Envelope(minx, maxx, miny, maxy)
      opts.set(SpatialFileRDD.FilterMBR, Array(minx, miny, maxx, maxy).mkString(","))
    } catch {
      case e: NullPointerException =>
    }
    // MBR not passed. Use all farmlands
    if (soilDepth == null || layer == null) {
      val writer = response.getWriter
      writer.printf("{\"error\": \"Error! Both 'soildepth' and 'layer' parameters are required\"}")
      return true
    }
    // Initialize the reader that reads the relevant farmlands// Initialize the reader that reads the relevant farmlands
    reader.initialize(inputFileSplit, opts)
    // Retrieve in an array to prepare the zonal statistics calculation// Retrieve in an array to prepare the zonal statistics calculation
    val farmlands = reader.asScala.toArray
    reader.close()
    logInfo(s"Read ${farmlands.length} records in ${(System.nanoTime() - t1) *1E-9} seconds")

    // load raster data based on selected soil depth and layer// load raster data based on selected soil depth and layer
    val matchingRasterDirs: Array[String] = rasterFiles
      .filter(rasterFile => SoilServlet.rangeOverlap(rasterFile._1, soilDepth))
      .map(rasterFile => s"data/POLARIS/$layer/${rasterFile._2}")
      .toArray

    val matchingRasterFiles = if (mbr != null) {
      val fileSystem = new Path(dataPath).getFileSystem(sparkSession.sparkContext.hadoopConfiguration)
      val geom = new GeometryFactory().toGeometry(mbr)
      geom.setSRID(4326)
      matchingRasterDirs.flatMap(matchingRasterDir =>
        RasterFileRDD.selectFiles(fileSystem, matchingRasterDir, geom))
    } else {
      matchingRasterDirs
    }

    if (System.nanoTime() - t1 > ServerTimeout) {
      sendTimeout(response, System.nanoTime() - t1)
      return true
    }

    logDebug(s"Query matched ${matchingRasterFiles.length} files")

    // Load raster data// Load raster data
    val finalResults: Iterator[(Int, SingleMachineRaptorJoin.Statistics)] =
      SingleMachineRaptorJoin.zonalStatistics(matchingRasterFiles, farmlands.map(_.getGeometry),
        () => System.nanoTime() - t1 > ServerTimeout)

    if (System.nanoTime() - t1 > ServerTimeout) {
      sendTimeout(response, System.nanoTime() - t1)
      return true
    }

    // write results to json object// write results to json object
    val out = response.getWriter
    val mapper = new ObjectMapper

    // create query node// create query node
    val queryNode = mapper.createObjectNode
    queryNode.put("soildepth", soilDepth)
    queryNode.put("layer", layer)

    // create mbr node// create mbr node
    // inside query node// inside query node
    if (mbr != null) {
      val mbrNode = mapper.createObjectNode
      mbrNode.put("minx", mbr.getMinX)
      mbrNode.put("miny", mbr.getMinY)
      mbrNode.put("maxx", mbr.getMaxX)
      mbrNode.put("maxy", mbr.getMaxY)
      queryNode.set("mbr", mbrNode)
    }

    // create results node// create results node
    val resultsNode = mapper.createArrayNode

    // populate json object with max vals
    for ((i, s) <- finalResults; if s != null) {
      val farmland = farmlands(i)
      val resultNode = mapper.createObjectNode
      resultNode.put("objectid", farmland.getAs("OBJECTID").asInstanceOf[Number].longValue)
      resultNode.put("min", s.min)
      resultNode.put("max", s.max)
      resultNode.put("average", s.mean)
      resultNode.put("count", s.count)
      resultNode.put("stdev", s.stdev)
      resultNode.put("median", s.median)
      resultsNode.add(resultNode)
      if (System.nanoTime() - t1 > ServerTimeout) {
        sendTimeout(response, System.nanoTime() - t1)
        return true
      }
    }

    // create root node// create root node
    // contains queryNode and resultsNode// contains queryNode and resultsNode
    val rootNode = mapper.createObjectNode
    rootNode.set("query", queryNode)
    rootNode.set("results", resultsNode)

    // write values to response writer// write values to response writer
    val jsonString = mapper.writer.writeValueAsString(rootNode)
    out.print(jsonString)
    out.flush()

    logInfo(s"multipolygon processing took ${(System.nanoTime() - t1)*1E-9} seconds")
    true
  }

  @WebMethod(url = "/soil/image.png", order = 1)
  def soilImage(path: String, request: HttpServletRequest, response: HttpServletResponse): Boolean = {
    val t1 = System.nanoTime()
    // sidebar select parameters
    var soilDepth = ""
    var layer = ""

    // try getting parameters from url
    try {
      soilDepth = request.getParameter("soildepth")
      layer = request.getParameter("layer")
    } catch {
      case e: NullPointerException => throw new RuntimeException("Couldn't find the required parameters: soildepth and layer")
    }

    // load raster data based on selected soil depth and layer// load raster data based on selected soil depth and layer
    val matchingRasterDirs: Array[String] = rasterFiles
      .filter(rasterFile => SoilServlet.rangeOverlap(rasterFile._1, soilDepth))
      .map(rasterFile => s"data/POLARIS/$layer/${rasterFile._2}")
      .toArray

    val baos = new ByteArrayOutputStream
    val input = request.getInputStream
    IOUtils.copy(input, baos)
    input.close()
    baos.close()
    val geoJSONData: Array[Byte] = baos.toByteArray
    var geom: Geometry = null
    try {
      val jsonParser = new JsonFactory().createParser(new ByteArrayInputStream(geoJSONData))
      geom = GeoJSONFeatureReader.parseGeometry(jsonParser)
      geom.setSRID(4326)
      jsonParser.close()
    } catch {
      case e: Exception =>
        logError(s"Error parsing GeoJSON geometry ${new String(geoJSONData)}", e)
        throw new RuntimeException(s"Error parsing query geometry ${new String(geoJSONData)}", e)
    }

    // Now that we have a geometry object, call single machine raptor join to retrieve all pixels in the query polygon
    val fileSystem = new Path(dataPath).getFileSystem(sparkSession.sparkContext.hadoopConfiguration)
    val matchingFiles = matchingRasterDirs.flatMap(matchingRasterDir =>
      RasterFileRDD.selectFiles(fileSystem, matchingRasterDir, geom))
    logDebug(s"Query matched ${matchingFiles.length} files")
    if (System.nanoTime() - t1 > ServerTimeout) {
      sendTimeout(response, System.nanoTime() - t1)
      return true
    }

    val intersections: Array[(Int, Intersections)] = matchingFiles.zipWithIndex.map({ case (rasterFileName: String, index: Int) =>
      val rasterFS: FileSystem = new Path(rasterFileName).getFileSystem(new Configuration())
      val rasterReader = RasterHelper.createRasterReader(rasterFS, new Path(rasterFileName), new BeastOptions())
      val intersections = new Intersections()
      intersections.compute(Array(geom), rasterReader.metadata)
      rasterReader.close()
      if (System.nanoTime() - t1 > ServerTimeout) {
        sendTimeout(response, System.nanoTime() - t1)
        return true
      }
      (index, intersections)
    }).filter(_._2.getNumIntersections > 0)
    if (intersections.isEmpty)
      return false
    val intersectionIterator: Iterator[(Long, PixelRange)] = new IntersectionsIterator(intersections.map(_._1), intersections.map(_._2))
    val pixels: Iterator[RaptorJoinResult[Float]] = new PixelIterator[Float](intersectionIterator, matchingFiles, "0")

    // Arrange the pixels in an image
    val resolution = 256
    val sums = new Array[Float](resolution * resolution)
    val counts = new Array[Int](resolution * resolution)
    val imageMBR = Reprojector.reprojectEnvelope(geom.getEnvelopeInternal, 4326, 3857)
    val imageMetadata = RasterMetadata.create(imageMBR.getMinX, imageMBR.getMaxY, imageMBR.getMaxX, imageMBR.getMinY,
      3857, resolution, resolution, resolution, resolution)

    val cachedTransformations: scala.collection.mutable.HashMap[RasterMetadata, MathTransform] =
      scala.collection.mutable.HashMap.empty[RasterMetadata, MathTransform]

    for (pixel <- pixels) {
      if (System.nanoTime() - t1 > ServerTimeout) {
        sendTimeout(response, System.nanoTime() - t1)
        return true
      }
      val pixelMBR = Array[Double](pixel.x, pixel.y,
        pixel.x + 1, pixel.y,
        pixel.x + 1, pixel.y + 1,
        pixel.x, pixel.y + 1,
      )
      val transformation = cachedTransformations.getOrElseUpdate(pixel.rasterMetadata, {
        val t1 = new AffineTransform2D(pixel.rasterMetadata.g2m)
        val t2 = Reprojector.findTransformationInfo(pixel.rasterMetadata.srid, 3857).mathTransform
        val t3 = new AffineTransform2D(imageMetadata.g2m.createInverse())
        ConcatenatedTransform.create(t1, ConcatenatedTransform.create(t2, t3))
      })
      transformation.transform(pixelMBR, 0, pixelMBR, 0, 4)
      val minX: Int = 0 max (pixelMBR(0).round min pixelMBR(2).round min pixelMBR(4).round min pixelMBR(6).round).toInt
      val maxX: Int = resolution min (pixelMBR(0).round max pixelMBR(2).round max pixelMBR(4).round max pixelMBR(6).round).toInt
      val minY: Int = 0 max (pixelMBR(1).round min pixelMBR(3).round min pixelMBR(5).round min pixelMBR(7).round).toInt
      val maxY: Int = resolution min (pixelMBR(1).round max pixelMBR(3).round max pixelMBR(5).round max pixelMBR(7).round) .toInt
      for (y <- minY until maxY; x <- minX until maxX) {
        val offset = y * resolution + x
        sums(offset) += pixel.m
        counts(offset) += 1
      }
    }
    // Convert the array of values to an image
    val averages = counts.zip(sums).map(x => if(x._1 == 0) Float.NaN else x._2 / x._1)
    val minM = averages.filterNot(_.isNaN).min
    val maxM = averages.filterNot(_.isNaN).max

    if (System.nanoTime() - t1 > ServerTimeout) {
      sendTimeout(response, System.nanoTime() - t1)
      return true
    }

    val targetImage = new BufferedImage(resolution, resolution, BufferedImage.TYPE_INT_ARGB)
    for (offset <- averages.indices; if counts(offset) > 0) {
      val x = offset % resolution
      val y = offset / resolution
      val pixelValue = averages(offset)
      val scale: Int = ((pixelValue - minM) * 255 / (maxM - minM)).toInt
      val color = new Color(scale, scale, scale)
      targetImage.setRGB(x, y, color.getRGB)
    }
    // Return the resulting image
    // set content-type as application/json
    response.setContentType("image/png")
    response.setStatus(HttpServletResponse.SC_OK)
    val out = response.getOutputStream
    ImageIO.write(targetImage, "png", out)
    out.close()
    logInfo(s"image generation took ${(System.nanoTime() - t1)*1E-9} seconds")
    true
  }
}

object SoilServlet {
  /** Maximum processing time on the server for interactive requests in nano seconds */
  val ServerTimeout: Long = 10E9.toLong

  def rangeOverlap(r1: String, r2: String): Boolean = {
    val parts1 = r1.split("-")
    val begin1 = parts1(0).toInt
    val end1 = parts1(1).toInt
    val parts2 = r2.split("-")
    val begin2 = parts2(0).toInt
    val end2 = parts2(1).toInt
    // Treat both ranges as begin-inclusive and end-exclusive
    !(begin1 >= end2 || begin2 >= end1)
  }

  val rasterFiles: Map[String, String] = Map(
    "0-5" -> "0_5_compressed",
    "5-15" -> "5_15_compressed",
    "15-30" -> "15_30_compressed",
    "30-60" -> "30_60_compressed",
    "60-100" -> "60_100_compressed",
    "100-200" -> "100_200_compressed"
  )

  def sendTimeout(response: HttpServletResponse, timeout: Long): Unit = {
    response.setStatus(HttpServletResponse.SC_SERVICE_UNAVAILABLE)
    response.setContentType("application/json")
    val writer = response.getWriter
    writer.println(s"""{"message": "Request is too costly", "time": ${timeout*1E-9}}""")
    writer.close()
  }
}
