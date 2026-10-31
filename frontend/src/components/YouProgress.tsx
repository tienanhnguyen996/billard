import React from 'react';

interface YouProgressProps {
  onInspectSession: () => void;
  onFilterHighlights: (filter: string) => void;
}

export const YouProgress: React.FC<YouProgressProps> = ({ onInspectSession, onFilterHighlights }) => {
  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)' }}>
            You &mdash; Progress &amp; Sessions
          </h2>
          <p style={{ fontSize: '0.82rem', color: 'var(--mut)' }}>
            AI Shot-by-Shot Performance Tracking &bull; Verdict Breakdown &bull; Weekly Trends
          </p>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="rb-btn active">📈 Progress</button>
          <button className="rb-btn" onClick={onInspectSession}>🎞️ Sessions (2)</button>
        </div>
      </div>

      {/* Highlights 4-Card Row */}
      <div>
        <div style={{ fontSize: '0.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--mut)', marginBottom: '0.5rem' }}>
          Highlights
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.85rem' }}>
          <div
            className="rb-card"
            style={{ padding: '0.9rem', cursor: 'pointer', borderLeft: '3px solid var(--volt)' }}
            onClick={() => onFilterHighlights('best')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 700, color: 'var(--bone)', fontSize: '0.92rem' }}>⭐ Favorites</span>
              <span className="rb-glyph" style={{ background: '#1BAAA0' }}>!!</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.4rem', fontWeight: 800, color: 'var(--volt)', marginTop: '0.4rem' }}>
              5 Clips
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--mut)' }}>Ginned &amp; Highlight Shots &rarr; Watch</div>
          </div>

          <div
            className="rb-card"
            style={{ padding: '0.9rem', cursor: 'pointer', borderLeft: '3px solid #2F8EF5' }}
            onClick={() => onFilterHighlights('make')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 700, color: 'var(--bone)', fontSize: '0.92rem' }}>🔥 Difficult Makes</span>
              <span className="rb-glyph" style={{ background: '#2F8EF5' }}>!</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.4rem', fontWeight: 800, color: '#2F8EF5', marginTop: '0.4rem' }}>
              8 Clips
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--mut)' }}>Cut &gt; 25&deg; or Distance &gt; 1.8m &rarr; Watch</div>
          </div>

          <div
            className="rb-card"
            style={{ padding: '0.9rem', cursor: 'pointer', borderLeft: '3px solid #D64545' }}
            onClick={() => onFilterHighlights('all')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 700, color: 'var(--bone)', fontSize: '0.92rem' }}>⚠️ Easy Misses</span>
              <span className="rb-glyph" style={{ background: '#D64545' }}>??</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.4rem', fontWeight: 800, color: '#D64545', marginTop: '0.4rem' }}>
              0 Clips
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--mut)' }}>Dogged / Inaccurate Leaves &rarr; Review</div>
          </div>

          <div
            className="rb-card"
            style={{ padding: '0.9rem', cursor: 'pointer', borderLeft: '3px solid var(--spring)' }}
            onClick={onInspectSession}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 700, color: 'var(--bone)', fontSize: '0.92rem' }}>⚡ Runs &amp; Streaks</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--spring)', fontWeight: 700 }}>6-PACK</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.4rem', fontWeight: 800, color: 'var(--spring)', marginTop: '0.4rem' }}>
              6-Ball Run
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--mut)' }}>Consecutive Runout Streak &rarr; Replay</div>
          </div>
        </div>
      </div>

      {/* This Week Activity & Step-Area SVG Chart */}
      <div className="rb-card" style={{ padding: '1.1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--bone)' }}>
            📅 This Week &mdash; Activity &amp; Volume
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--volt)' }}>
            +24% vs Last Week
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Shots Tracked</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 700, color: 'var(--bone)' }}>284</div>
          </div>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Time Played</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 700, color: 'var(--bone)' }}>3h 42m</div>
          </div>
          <div>
            <div style={{ fontSize: '0.7rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Sessions</div>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 700, color: 'var(--volt)' }}>6</div>
          </div>
        </div>

        {/* Step-area chart */}
        <svg viewBox="0 0 600 95" style={{ width: '100%', height: 90, display: 'block' }}>
          <defs>
            <linearGradient id="rbWeekGradReact" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#FEF84C" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#FEF84C" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path d="M0,75 L85,75 L85,52 L170,52 L170,60 L255,60 L255,34 L340,34 L340,42 L425,42 L425,18 L510,18 L510,12 L600,12 L600,90 L0,90 Z" fill="url(#rbWeekGradReact)" />
          <path d="M0,75 L85,75 L85,52 L170,52 L170,60 L255,60 L255,34 L340,34 L340,42 L425,42 L425,18 L510,18 L510,12 L600,12" fill="none" stroke="#FEF84C" strokeWidth="2.5" />
        </svg>
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.85rem' }}>
        <div className="rb-card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Total Performance Average (TPA)</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.8rem', fontWeight: 800, color: 'var(--volt)', marginTop: 4 }}>0.985</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--spring)' }}>World-Class Pro Grade (Hoàng Sao)</div>
        </div>

        <div className="rb-card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Potting Success Rate</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.8rem', fontWeight: 800, color: 'var(--spring)', marginTop: 4 }}>100.0%</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)' }}>11 Made / 11 Total in Match</div>
        </div>

        <div className="rb-card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Average Cue Ball Speed</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.8rem', fontWeight: 800, color: 'var(--bone)', marginTop: 4 }}>15.4 km/h</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--cyan)' }}>Max: 23.8 km/h (Break Gear 4)</div>
        </div>

        <div className="rb-card" style={{ padding: '1rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Average Shot Difficulty</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.8rem', fontWeight: 800, color: 'var(--bone)', marginTop: 4 }}>48.2 / 100</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--mut)' }}>Grade B (Tour Standard)</div>
        </div>
      </div>
    </div>
  );
};
