/**
 * Main dashboard layout with sidebar
 */
import React from 'react';
import { Activity, BarChart3, Settings, Container } from 'lucide-react';
import { TradingStatusIndicator } from './TradingStatusIndicator';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="w-64 bg-dark-card border-r border-dark-border flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-dark-border">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="text-profit" />
            <span>Adaptive Bot</span>
          </h1>
          <p className="text-sm text-dark-text-secondary mt-1">Trading Dashboard</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            <NavItem icon={<BarChart3 size={20} />} label="Overview" active />
            <NavItem icon={<Activity size={20} />} label="Active Trades" />
            <NavItem icon={<Container size={20} />} label="Containers" />
            <NavItem icon={<Settings size={20} />} label="Settings" />
          </ul>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-dark-border space-y-3">
          {/* API Status */}
          <div className="flex items-center gap-2 text-sm text-dark-text-secondary">
            <div className="w-2 h-2 rounded-full bg-profit animate-pulse-slow"></div>
            <span>API Connected</span>
          </div>

          {/* Trading Status */}
          <TradingStatusIndicator />
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-dark-bg">
        <div className="p-8">
          {children}
        </div>
      </main>
    </div>
  );
};

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, active }) => {
  return (
    <li>
      <button
        className={`
          w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors
          ${active
            ? 'bg-blue-600 text-white'
            : 'text-dark-text-secondary hover:bg-dark-border hover:text-dark-text-primary'
          }
        `}
      >
        {icon}
        <span className="font-medium">{label}</span>
      </button>
    </li>
  );
};
