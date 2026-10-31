import React from 'react';
import { Shot } from '../types';
import { getShotVerdict } from '../utils/geometry';

interface PlayerAiStatsRailProps {
  shot?: Shot;
  playerName?: string;
  playerRating?: string;
}

export const PlayerAiStatsRail: React.FC<PlayerAiStatsRailProps> = ({
  shot,
  playerName = 'Dương Quốc Hoàng',
  playerRating = '98.5'
}) => {
  const verdict = getShotVerdict(shot);
  const isBreak = shot?.shot_number === 1;

  const outUp = (shot?.outcome || '').toUpperCase();
  const ratingScore = outUp.includes('MAKE')
    ? Math.min(99, Math.round(72 + Number(shot?.difficulty_score || 45) * 0.28))
    : Math.max(28, Math.round(55 - Number(shot?.difficulty_score || 45) * 0.25));

  return (
    <aside className="rb-ai-rail">
      {/* Player Header */}
      <div>
        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: 'var(--bone)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {playerName}
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 2 }}>
          <span className="rb-rail-hdr">RATING</span>
          <span className="rb-rail-val volt" style={{ fontSize: '1.05rem' }}>{playerRating}</span>
        </div>
      </div>

      <div style={{ height: 1, background: 'var(--panel-border)' }} />

      {/* Current Shot Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
        <div className="rb-rail-hdr">
          CURRENT SHOT &bull; #{shot?.shot_number || 1}
          {isBreak && <span style={{ color: 'var(--cyan)', marginLeft: 4 }}>BREAK</span>}
        </div>

        <div className="rb-rail-metric">
          <span className="rb-rail-hdr">SPEED</span>
          <span className="rb-rail-val">{shot?.max_speed_kmh || 15.4} km/h</span>
        </div>

        <div className="rb-rail-metric">
          <span className="rb-rail-hdr">CUT ANGLE</span>
          <span className="rb-rail-val cyan">{shot?.cut_angle_deg || 0}&deg;</span>
        </div>

        <div className="rb-rail-metric">
          <span className="rb-rail-hdr">SPIN</span>
          <span className="rb-rail-val" style={{ fontSize: '0.8rem' }}>
            {shot?.cue_tip?.spin_name || 'Center Ball'}
          </span>
        </div>

        <div className="rb-rail-metric">
          <span className="rb-rail-hdr">MARGIN</span>
          <span className="rb-rail-val spring">&plusmn;{shot?.margin_deg || 2.1}&deg;</span>
        </div>

        <div className="rb-rail-metric" style={{ borderTop: '1px solid var(--panel-border)', paddingTop: '0.4rem' }}>
          <span className="rb-rail-hdr">SHOT RATING</span>
          <span className="rb-rail-val volt">{ratingScore} / 100</span>
        </div>
      </div>

      {/* Chess.com Verdict Badge */}
      <div className="rb-verdict-box" title={`${verdict.label}: ${verdict.description}`}>
        <div className="rb-glyph" style={{ background: verdict.background }}>
          {verdict.glyph}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--bone)' }}>
            {verdict.label}
          </span>
          <span style={{ fontSize: '0.65rem', color: 'var(--mut)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {verdict.description}
          </span>
        </div>
      </div>
    </aside>
  );
};
