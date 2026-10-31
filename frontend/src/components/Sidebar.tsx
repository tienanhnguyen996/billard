import React from 'react';
import {
  Video,
  Home,
  User,
  Users,
  UploadCloud,
  Radio,
  Target,
  FlaskConical,
  Trophy,
  ShieldCheck
} from 'lucide-react';

interface SidebarProps {
  currentView: string;
  onSelectView: (view: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onSelectView }) => {
  const navItems = [
    { id: 'match', label: 'Player Studio', icon: Video },
    { id: 'home', label: 'Home', icon: Home },
    { id: 'analytics', label: 'You', icon: User },
    { id: 'following', label: 'Following', icon: Users },
    { id: 'upload', label: 'Upload', icon: UploadCloud, badge: 'Early access' },
    { id: 'record', label: 'Record', icon: Radio, badge: 'Early access' },
    { id: 'drills', label: 'Drills', icon: Target },
    { id: 'shotlab', label: 'Shot Lab', icon: FlaskConical },
    { id: 'leaderboards', label: 'Leaderboards', icon: Trophy },
    { id: 'qa', label: 'QA & Benchmarks', icon: ShieldCheck }
  ];

  return (
    <aside className="rb-sidebar">
      <div className="rb-sidebar-brand" onClick={() => onSelectView('match')}>
        <img src="/railbird-mark.png" alt="Railbird" onError={(e) => { (e.target as HTMLElement).style.display = 'none'; }} />
        <span className="rb-sidebar-brand-title">railbird</span>
      </div>

      <nav className="rb-nav-menu">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              className={`rb-nav-link ${isActive ? 'active' : ''}`}
              onClick={() => onSelectView(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {item.badge && <span className="rb-early-badge">{item.badge}</span>}
            </button>
          );
        })}
      </nav>

      <div className="rb-sidebar-user">
        <div className="rb-avatar">HS</div>
        <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--bone)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            Hoàng Sao &bull; Pro
          </span>
          <span style={{ fontSize: '0.68rem', color: 'var(--volt)', fontFamily: 'var(--font-mono)' }}>
            TPA 0.985 &bull; Fargo 818
          </span>
        </div>
      </div>
    </aside>
  );
};
