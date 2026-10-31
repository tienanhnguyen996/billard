import React, { useEffect, useRef } from 'react';
import { Shot } from '../types';
import { drawMiniShotDiagram, getShotTiming, getShotVerdict } from '../utils/geometry';

interface ShotsPlaylistProps {
  shots: Shot[];
  activeShotIndex: number;
  shotFilter: string;
  onFilterChange: (filter: string) => void;
  onSelectShot: (index: number) => void;
  onPracticeInLab: (shot: Shot) => void;
}

const MiniDiagramCanvas: React.FC<{ shot: Shot }> = ({ shot }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (canvasRef.current) {
      drawMiniShotDiagram(canvasRef.current, shot, []);
    }
  }, [shot]);

  return <canvas ref={canvasRef} className="rb-mini-canvas" width={280} height={128} />;
};

export const ShotsPlaylist: React.FC<ShotsPlaylistProps> = ({
  shots,
  activeShotIndex,
  shotFilter,
  onFilterChange,
  onSelectShot,
  onPracticeInLab
}) => {
  let displayShots = shots;
  if (shotFilter === 'make') {
    displayShots = shots.filter(s => (s.outcome || '').toUpperCase().includes('MAKE'));
  } else if (shotFilter === 'safety') {
    displayShots = shots.filter(s => (s.outcome || '').toUpperCase().includes('SAFETY'));
  } else if (shotFilter === 'best') {
    displayShots = shots.filter(s => s.is_highlight);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--bone)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
          ⚡ Railbird Shot Feed
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--volt)' }}>
          {displayShots.length} Shots
        </span>
      </div>

      {/* Filter Tabs */}
      <div style={{ display: 'flex', gap: 4, background: 'var(--ink-2)', padding: 3, borderRadius: 6, border: '1px solid var(--panel-border)' }}>
        {['all', 'make', 'safety', 'best'].map(f => (
          <button
            key={f}
            className={`rb-pill-btn ${shotFilter === f ? 'active' : ''}`}
            onClick={() => onFilterChange(f)}
            style={{ flex: 1, textAlign: 'center', textTransform: 'capitalize' }}
          >
            {f === 'best' ? '⭐ Best' : f}
          </button>
        ))}
      </div>

      {/* Shot Cards List */}
      <div className="rb-shots-list">
        {displayShots.map((shot) => {
          const originalIdx = shots.findIndex(s => s.shot_number === shot.shot_number);
          const isActive = originalIdx === activeShotIndex;
          const verdict = getShotVerdict(shot);
          const { strokeT } = getShotTiming(shot);
          const tsMin = Math.floor(strokeT / 60);
          const tsSec = String(Math.floor(strokeT % 60)).padStart(2, '0');
          const isBreak = shot.shot_number === 1;

          return (
            <div
              key={shot.shot_number}
              className={`rb-shot-card ${isActive ? 'active' : ''}`}
              onClick={() => onSelectShot(originalIdx)}
            >
              {/* Card Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--bone)' }}>
                  Shot #{shot.shot_number} &bull; {shot.target_ball || 'Ball'}
                  {shot.is_highlight && <span style={{ color: 'var(--volt)', marginLeft: 6, fontSize: '0.7rem' }}>⭐ BEST</span>}
                </span>
                <span className="rb-glyph" style={{ background: verdict.background }}>
                  {verdict.glyph}
                </span>
              </div>

              {/* Mini 2D Vector Shot Diagram */}
              <div style={{ position: 'relative' }}>
                <MiniDiagramCanvas shot={shot} />
                <div style={{
                  position: 'absolute',
                  bottom: 6,
                  left: 12,
                  right: 12,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.72rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--bone)',
                  pointerEvents: 'none'
                }}>
                  <span style={{ color: verdict.background, fontWeight: 700 }}>{verdict.label}</span>
                  <span>{tsMin}:{tsSec}</span>
                  {isBreak ? (
                    <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>BREAK</span>
                  ) : (
                    <span style={{ color: 'var(--volt)', fontWeight: 700 }}>{shot.cut_angle_deg || 0}&deg;</span>
                  )}
                </div>
              </div>

              {/* Metrics */}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--mut)', fontFamily: 'var(--font-mono)' }}>
                <span>Cut: <strong>{shot.cut_angle_deg || 0}&deg;</strong></span>
                <span>Margin: <strong>&plusmn;{shot.margin_deg || 2.1}&deg;</strong></span>
                <span>Diff: <strong>{shot.difficulty_score || 40}/100</strong></span>
                <span>Speed: <strong>{shot.max_speed_kmh || 15} km/h</strong></span>
              </div>

              {/* Coach Tip Quote */}
              {shot.coach_tip && (
                <div style={{ fontStyle: 'italic', fontSize: '0.73rem', color: '#cbd5e1', borderLeft: '2px solid var(--volt)', paddingLeft: 8 }}>
                  &ldquo;{shot.coach_tip}&rdquo;
                </div>
              )}

              {/* Actions */}
              <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                <button
                  className="rb-btn"
                  style={{ flex: 1, justifyContent: 'center' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    onPracticeInLab(shot);
                  }}
                >
                  🎯 Practice in Lab
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
