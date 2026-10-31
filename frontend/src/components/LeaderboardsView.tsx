import React from 'react';

export const LeaderboardsView: React.FC = () => {
  const rows = [
    { rank: 1, name: 'Fedor Gorst', spec: '9-ft • 4.0"', sessions: 64, make: '94.0%', run: '11-Pack', ginned: '42%', rating: '99.2' },
    { rank: 2, name: 'Dương Quốc Hoàng (Hoàng Sao)', spec: '9-ft • 4.0"', sessions: 48, make: '92.3%', run: '9-Pack', ginned: '38%', rating: '98.5' },
    { rank: 3, name: 'Francisco Sánchez Ruiz', spec: '9-ft • 4.0"', sessions: 53, make: '91.8%', run: '9-Pack', ginned: '36%', rating: '98.1' },
    { rank: 4, name: 'Joshua Filler', spec: '9-ft • 4.0"', sessions: 41, make: '91.2%', run: '8-Pack', ginned: '35%', rating: '97.8' },
    { rank: 5, name: 'Dean Wenstrand', spec: '9-ft • 4.5"', sessions: 112, make: '69.1%', run: '7-Ball', ginned: '18%', rating: '84.6' }
  ];

  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)' }}>
          Global &amp; Club Leaderboards
        </h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--mut)' }}>
          Ranked by Verified Railbird TPA, Make %, and Longest Runout Streak
        </p>
      </div>

      <div className="rb-card" style={{ padding: '0.5rem', overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--panel-border)', color: 'var(--mut)' }}>
              <th style={{ padding: '0.75rem' }}>Rank</th>
              <th style={{ padding: '0.75rem' }}>Player</th>
              <th style={{ padding: '0.75rem' }}>Table Spec</th>
              <th style={{ padding: '0.75rem' }}>Sessions</th>
              <th style={{ padding: '0.75rem' }}>Make Rate</th>
              <th style={{ padding: '0.75rem' }}>High Run</th>
              <th style={{ padding: '0.75rem' }}>Ginned (!!)</th>
              <th style={{ padding: '0.75rem' }}>Rating</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.rank} style={{ borderBottom: '1px solid var(--panel-border)' }}>
                <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', fontWeight: 700, color: r.rank <= 3 ? 'var(--volt)' : 'var(--bone)' }}>#{r.rank}</td>
                <td style={{ padding: '0.75rem', fontWeight: 700, color: 'var(--bone)' }}>{r.name}</td>
                <td style={{ padding: '0.75rem', color: 'var(--mut)', fontFamily: 'var(--font-mono)' }}>{r.spec}</td>
                <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)' }}>{r.sessions}</td>
                <td style={{ padding: '0.75rem', color: 'var(--spring)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{r.make}</td>
                <td style={{ padding: '0.75rem' }}><span style={{ background: 'rgba(254, 248, 76, 0.15)', color: 'var(--volt)', padding: '2px 8px', borderRadius: 4, fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 700 }}>{r.run}</span></td>
                <td style={{ padding: '0.75rem', color: '#1BAAA0', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{r.ginned}</td>
                <td style={{ padding: '0.75rem', color: 'var(--volt)', fontWeight: 800, fontFamily: 'var(--font-mono)', fontSize: '0.95rem' }}>{r.rating}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
