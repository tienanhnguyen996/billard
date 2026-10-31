import React from 'react';
import { Shot } from '../types';
import { getShotTiming, getShotVerdict } from '../utils/geometry';

interface ShotTimelineProps {
  shots: Shot[];
  activeShotIndex: number;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
  onSelectShot: (index: number) => void;
}

export const ShotTimeline: React.FC<ShotTimelineProps> = ({
  shots,
  activeShotIndex,
  duration,
  currentTime,
  onSeek,
  onSelectShot
}) => {
  const dur = duration > 0 ? duration : 60;
  const playheadPct = Math.max(0, Math.min(100, (currentTime / dur) * 100));

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(ratio * dur);
  };

  return (
    <div className="rb-timeline-wrap" onClick={handleClick} title="Click anywhere to scrub playback">
      <div className="rb-timeline-track">
        {shots.map((shot, idx) => {
          const { strokeT } = getShotTiming(shot);
          const pct = Math.max(1, Math.min(99, (strokeT / dur) * 100));
          const verdict = getShotVerdict(shot);
          const isActive = idx === activeShotIndex;

          return (
            <div
              key={shot.shot_number || idx}
              className={`rb-timeline-tick ${isActive ? 'active' : ''}`}
              style={{
                left: `${pct}%`,
                background: verdict.background
              }}
              title={`Shot #${shot.shot_number}: ${shot.target_ball || 'Ball'} (${verdict.label} - ${shot.outcome})`}
              onClick={(e) => {
                e.stopPropagation();
                onSelectShot(idx);
              }}
            />
          );
        })}

        <div className="rb-timeline-playhead" style={{ left: `${playheadPct}%` }} />
      </div>
    </div>
  );
};
