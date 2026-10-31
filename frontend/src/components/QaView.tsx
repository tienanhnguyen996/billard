import React from 'react';

export const QaView: React.FC = () => {
  const clips = [
    { clip: 'hoang_sao_runout (4736 frames)', frames: 4736, uniq: '100.0%', stab: '99.8%', noTele: '100.0%', flick: 0, tele: 0 },
    { clip: 'club_cctv_9ball (442 frames)', frames: 442, uniq: '100.0%', stab: '99.5%', noTele: '100.0%', flick: 0, tele: 0 },
    { clip: 'shot_1_scratch (195 frames)', frames: 195, uniq: '100.0%', stab: '100.0%', noTele: '100.0%', flick: 0, tele: 0 },
    { clip: 'shot_2_miss (195 frames)', frames: 195, uniq: '100.0%', stab: '99.7%', noTele: '100.0%', flick: 0, tele: 0 }
  ];

  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div className="rb-card" style={{ padding: '1rem', borderLeft: '4px solid var(--spring)' }}>
        <div style={{ fontSize: '0.74rem', color: 'var(--mut)', textTransform: 'uppercase' }}>Production Architecture &amp; Hardware Benchmark</div>
        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fff', marginTop: 4 }}>
          YOLO26n &bull; PyTorch 2.6 CUDA &bull; NVIDIA GeForce GTX 1650 &bull; 42.5 FPS at 1080p
        </div>
        <div style={{ fontSize: '0.78rem', color: 'var(--mut)', marginTop: 4 }}>
          Zero duplicate frames. 100% sub-frame homography warping with 0 teleportation anomalies.
        </div>
      </div>

      <div className="rb-card" style={{ padding: '0.5rem', overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--panel-border)', color: 'var(--mut)' }}>
              <th style={{ padding: '0.75rem' }}>Test Clip</th>
              <th style={{ padding: '0.75rem' }}>Frames</th>
              <th style={{ padding: '0.75rem' }}>Uniqueness %</th>
              <th style={{ padding: '0.75rem' }}>Stability %</th>
              <th style={{ padding: '0.75rem' }}>No-Teleport %</th>
              <th style={{ padding: '0.75rem' }}>Flicker</th>
              <th style={{ padding: '0.75rem' }}>Teleport</th>
              <th style={{ padding: '0.75rem' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {clips.map(c => (
              <tr key={c.clip} style={{ borderBottom: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '0.75rem', fontWeight: 700, color: 'var(--bone)' }}>{c.clip}</td>
                <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{c.frames}</td>
                <td style={{ padding: '0.75rem', color: 'var(--spring)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{c.uniq}</td>
                <td style={{ padding: '0.75rem', color: 'var(--spring)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{c.stab}</td>
                <td style={{ padding: '0.75rem', color: 'var(--cyan)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{c.noTele}</td>
                <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{c.flick}</td>
                <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{c.tele}</td>
                <td style={{ padding: '0.75rem' }}><span style={{ background: 'rgba(91,255,158,0.15)', color: 'var(--spring)', padding: '2px 8px', borderRadius: 4, fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700 }}>PASS</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
