import React, { useState, useEffect } from 'react';
import { SessionData, Shot } from './types';
import { Sidebar } from './components/Sidebar';
import { PlayerStudio } from './components/PlayerStudio';
import { HomeFeed } from './components/HomeFeed';
import { YouProgress } from './components/YouProgress';
import { DrillsView, DrillDef } from './components/DrillsView';
import { ShotLab } from './components/ShotLab';
import { LeaderboardsView } from './components/LeaderboardsView';
import { QaView } from './components/QaView';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState('match');
  const [sessionMode, setSessionMode] = useState('hs');
  const [hsSession, setHsSession] = useState<SessionData | null>(null);
  const [clubSession, setClubSession] = useState<SessionData | null>(null);
  const [activeDrill, setActiveDrill] = useState<DrillDef | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/hoang_sao_session.json').then(r => r.json()),
      fetch('/session_data.json').then(r => r.json())
    ])
      .then(([hsData, clubData]) => {
        setHsSession(hsData);
        setClubSession(clubData);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load sessions from server:', err);
        // Fallback to window globals if served statically
        if ((window as any).HOANG_SAO_SESSION) {
          setHsSession((window as any).HOANG_SAO_SESSION);
        }
        if ((window as any).RAILBIRD_SESSION) {
          setClubSession((window as any).RAILBIRD_SESSION);
        }
        setLoading(false);
      });
  }, []);

  const activeSession = (sessionMode === 'hs' ? hsSession : clubSession) || hsSession;

  const handleLaunchDrill = (drill: DrillDef) => {
    setActiveDrill(drill);
    setCurrentView('shotlab');
  };

  const handlePracticeShotInLab = (shot: Shot) => {
    setActiveDrill({
      id: `shot_${shot.shot_number}`,
      title: `Practice Shot #${shot.shot_number} (${shot.target_ball})`,
      category: 'In-Match Replay',
      shots: 1,
      makePct: '100%',
      bestRun: 1,
      desc: shot.coach_tip || 'Replay exact match situation in the physics lab',
      cb: { rx: shot.aim_line?.cb_rx || 250, ry: shot.aim_line?.cb_ry || 250 },
      ob: { rx: shot.aim_line?.ob_rx || 700, ry: shot.aim_line?.ob_ry || 250 },
      pock: { rx: shot.aim_line?.pock_rx || 982, ry: shot.aim_line?.pock_ry || 18 },
      leave: { rx: 500, ry: 250 }
    });
    setCurrentView('shotlab');
  };

  if (loading || !activeSession) {
    return (
      <div style={{ width: '100vw', height: '100vh', background: 'var(--ink)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14 }}>
        <img src="/railbird-mark.png" alt="Railbird" style={{ width: 48, height: 48 }} />
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem', fontWeight: 700, color: 'var(--bone)' }}>
          Loading Railbird Match Studio...
        </div>
      </div>
    );
  }

  return (
    <div className="rb-shell">
      {/* Left Sidebar Drawer */}
      <Sidebar currentView={currentView} onSelectView={setCurrentView} />

      {/* Main Content Area */}
      <main className="rb-main">
        {currentView === 'match' && (
          <PlayerStudio
            session={activeSession}
            sessionMode={sessionMode}
            onSelectSession={setSessionMode}
            onPracticeInLab={handlePracticeShotInLab}
          />
        )}

        {currentView === 'home' && (
          <HomeFeed
            onOpenSession={(mode) => {
              setSessionMode(mode);
              setCurrentView('match');
            }}
            hsShot={hsSession?.shots[0]}
            clubShot={clubSession?.shots[0]}
          />
        )}

        {currentView === 'analytics' && (
          <YouProgress
            onInspectSession={() => setCurrentView('match')}
            onFilterHighlights={() => setCurrentView('match')}
          />
        )}

        {currentView === 'drills' && (
          <DrillsView onLaunchDrill={handleLaunchDrill} />
        )}

        {currentView === 'shotlab' && (
          <ShotLab activeDrill={activeDrill} />
        )}

        {currentView === 'leaderboards' && (
          <LeaderboardsView />
        )}

        {currentView === 'qa' && (
          <QaView />
        )}
      </main>
    </div>
  );
};
