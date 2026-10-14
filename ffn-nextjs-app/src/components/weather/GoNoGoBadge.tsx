'use client';

import React from 'react';
import { GoNoGoLevel, ScoreLevel } from '@/lib/types/weather';

interface GoNoGoBadgeProps {
  level: GoNoGoLevel;
  size?: 'sm' | 'md';
}

const GO_NO_GO_CONFIG: Record<GoNoGoLevel, { label: string; className: string }> = {
  go: {
    label: 'Go',
    className: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
  },
  caution: {
    label: 'Caution',
    className: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300',
  },
  'no-go': {
    label: 'No-Go',
    className: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  },
};

export function GoNoGoBadge({ level, size = 'sm' }: GoNoGoBadgeProps) {
  const config = GO_NO_GO_CONFIG[level];
  const sizeClass = size === 'sm' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-1';

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${config.className} ${sizeClass}`}
    >
      {config.label}
    </span>
  );
}

interface ScoreBadgeProps {
  level: ScoreLevel;
}

const SCORE_CONFIG: Record<ScoreLevel, { className: string }> = {
  good: { className: 'bg-green-500 dark:bg-green-600' },
  marginal: { className: 'bg-yellow-400 dark:bg-yellow-500' },
  bad: { className: 'bg-red-500 dark:bg-red-600' },
};

export function ScoreBlock({ level }: ScoreBadgeProps) {
  return (
    <div
      className={`w-full h-3 rounded-sm ${SCORE_CONFIG[level].className}`}
      title={level}
    />
  );
}
