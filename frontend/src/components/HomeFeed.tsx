import React from 'react';
import { ThumbsUp, MessageSquare, Play } from 'lucide-react';
import { Shot } from '../types';
import { drawMiniShotDiagram } from '../utils/geometry';

interface HomeFeedProps {
  onOpenSession: (mode: string) => void;
  hsShot?: Shot;
  clubShot?: Shot;
}

const FeedCanvasPreview: React.FC<{ shot?: Shot }> = ({ shot }) => {
  const ref = React.useRef<HTMLCanvasElement | null>(null);
  React.useEffect(() => {
    if (ref.current && shot) {
      drawMiniShotDiagram(ref.current, shot, []);
    }
  }, [shot]);
  return <canvas ref={ref} width={560} height={200} style={{ width: '100%', height: 160, borderRadius: 8, display: 'block', background: '#05070D' }} />;
};

export const HomeFeed: React.FC<HomeFeedProps> = ({ onOpenSession, hsShot, clubShot }) => {
  const feedItems = [
    {
      id: 'hs',
      initials: 'HS',
      player: 'Dương Quốc Hoàng',
      title: 'Matchroom European Open 9-Ball Runout',
      timeAgo: '2h ago • Fulda, Germany',
      tags: ['9-ft', '4.0"', 'Simonis 860 Shark Grey', 'Fargo 818'],
      shotsCount: 13,
      makeRate: '92.3%',
      duration: '4m 56s',
      likes: 148,
      comments: 29,
      verdictGlyph: '!!',
      verdictBg: '#1BAAA0',
      verdictText: '5 Ginned Shots • 6-Ball Runout Streak',
      shot: hsShot
    },
    {
      id: 'club',
      initials: 'DW',
      player: 'Dean Wenstrand',
      title: '9-Ball Ghost • 50 Shots Session',
      timeAgo: '5h ago • San Francisco, CA',
      tags: ['9-ft', '4.5"', 'Tournament Blue'],
      shotsCount: 55,
      makeRate: '69.1%',
      duration: '30m',
      likes: 42,
      comments: 8,
      verdictGlyph: '!',
      verdictBg: '#2F8EF5',
      verdictText: '12 Great Makes • High Run 7',
      shot: clubShot
    },
    {
      id: 'hs',
      initials: 'FG',
      player: 'Fedor Gorst',
      title: 'X Drill & Spot Shot Precision Session',
      timeAgo: '1d ago • Las Vegas, NV',
      tags: ['9-ft', '4.0"', 'Pro Cut Pockets'],
      shotsCount: 40,
      makeRate: '95.0%',
      duration: '22m',
      likes: 215,
      comments: 41,
      verdictGlyph: '!!',
      verdictBg: '#1BAAA0',
      verdictText: '19 Ginned Shots • 0.992 TPA',
      shot: hsShot
    }
  ];

  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)' }}>
          Home &mdash; Community Sessions Feed
        </h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--mut)' }}>
          Verified AI-Tracked Pool Sessions &bull; Click any session to inspect in Player Studio
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        {feedItems.map((item, idx) => (
          <div key={idx} className="rb-card" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div className="rb-avatar">{item.initials}</div>
                <div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--bone)' }}>
                    {item.player}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--mut)' }}>{item.timeAgo}</div>
                </div>
              </div>
              <div className="rb-glyph" style={{ background: item.verdictBg }}>{item.verdictGlyph}</div>
            </div>

            <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--volt)' }}>
              {item.title}
            </div>

            {/* Tags */}
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {item.tags.map(t => (
                <span key={t} style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', background: 'var(--ink-2)', border: '1px solid var(--panel-border)', padding: '2px 6px', borderRadius: 4, color: 'var(--mut)' }}>
                  {t}
                </span>
              ))}
              <span style={{ fontSize: '0.68rem', fontFamily: 'var(--font-mono)', background: 'rgba(91,255,158,0.1)', border: '1px solid rgba(91,255,158,0.3)', padding: '2px 6px', borderRadius: 4, color: 'var(--spring)' }}>
                {item.verdictText}
              </span>
            </div>

            {/* Stats */}
            <div style={{ display: 'flex', justifyContent: 'space-between', background: 'var(--ink-2)', padding: '0.6rem 0.85rem', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
              <div>Shots: <strong>{item.shotsCount}</strong></div>
              <div>Make Rate: <strong style={{ color: 'var(--spring)' }}>{item.makeRate}</strong></div>
              <div>Duration: <strong>{item.duration}</strong></div>
            </div>

            {/* Canvas Preview */}
            <div style={{ position: 'relative' }}>
              <FeedCanvasPreview shot={item.shot} />
              <button
                className="rb-btn active"
                style={{ position: 'absolute', bottom: 10, right: 10 }}
                onClick={() => onOpenSession(item.id)}
              >
                <Play size={13} /> Inspect Session
              </button>
            </div>

            {/* Footer */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.78rem', color: 'var(--mut)', paddingTop: 4 }}>
              <div style={{ display: 'flex', gap: 14 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                  <ThumbsUp size={13} /> {item.likes}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                  <MessageSquare size={13} /> {item.comments}
                </span>
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--cyan)' }}>YOLO26 Verified</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
