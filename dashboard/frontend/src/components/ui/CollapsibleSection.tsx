/**
 * Collapsible Section - Wrapper component for collapsible content
 */
import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultCollapsed?: boolean;
  badge?: string | number; // Optional badge (e.g., count)
  icon?: React.ReactNode;
}

export const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  defaultCollapsed = true,
  badge,
  icon,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <div>
      {/* Header Card - Clickable */}
      <div
        className="card cursor-pointer select-none hover:bg-dark-bg transition-colors"
        onClick={toggleCollapse}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Chevron Icon */}
            <div className="text-dark-text-secondary">
              {isCollapsed ? <ChevronRight size={20} /> : <ChevronDown size={20} />}
            </div>

            {/* Title */}
            <h2 className="text-xl font-bold text-dark-text-primary">{title}</h2>

            {/* Badge */}
            {badge !== undefined && (
              <span className="badge badge-primary text-sm">
                {badge}
              </span>
            )}

            {/* Icon */}
            {icon && <div className="text-dark-text-secondary">{icon}</div>}
          </div>

          {/* Collapse/Expand hint */}
          <span className="text-xs text-dark-text-secondary">
            {isCollapsed ? 'Click to expand' : 'Click to collapse'}
          </span>
        </div>
      </div>

      {/* Content - Collapsible */}
      {!isCollapsed && (
        <div className="mt-4 animate-fadeIn">
          {children}
        </div>
      )}
    </div>
  );
};
