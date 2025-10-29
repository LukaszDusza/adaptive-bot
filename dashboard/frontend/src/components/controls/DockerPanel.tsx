/**
 * Docker containers status and control panel
 */
import React from 'react';
import { Play, Square, RotateCw, Activity } from 'lucide-react';
import type { ContainerInfo } from '../../types';
import { startContainer, stopContainer, restartContainer } from '../../api/client';

interface DockerPanelProps {
  containers: ContainerInfo[];
  onRefresh?: () => void;
}

export const DockerPanel: React.FC<DockerPanelProps> = ({ containers, onRefresh }) => {
  const [loading, setLoading] = React.useState<string | null>(null);

  const handleAction = async (
    action: 'start' | 'stop' | 'restart',
    containerName: string
  ) => {
    setLoading(containerName);
    try {
      if (action === 'start') await startContainer(containerName);
      else if (action === 'stop') await stopContainer(containerName);
      else await restartContainer(containerName);

      // Refresh container list after action
      setTimeout(() => {
        onRefresh?.();
        setLoading(null);
      }, 1000);
    } catch (error) {
      console.error(`Failed to ${action} container:`, error);
      setLoading(null);
    }
  };

  const getStatusBadge = (status: string) => {
    const classes = {
      running: 'badge-success',
      stopped: 'badge-neutral',
      exited: 'badge-neutral',
      restarting: 'badge-warning',
      paused: 'badge-warning',
      dead: 'badge-danger',
    };
    return classes[status as keyof typeof classes] || 'badge-neutral';
  };

  const formatUptime = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Activity size={24} className="text-blue-500" />
          Docker Containers
        </h2>
        <button
          onClick={onRefresh}
          className="btn btn-secondary flex items-center gap-2"
        >
          <RotateCw size={16} />
          Refresh
        </button>
      </div>

      <div className="space-y-3">
        {containers.map((container) => (
          <div
            key={container.container_id}
            className="flex items-center justify-between p-4 bg-dark-bg rounded-lg border border-dark-border"
          >
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <span className="font-medium">{container.name}</span>
                <span className={`badge ${getStatusBadge(container.status)}`}>
                  {container.status}
                </span>
              </div>
              <div className="text-sm text-dark-text-secondary mt-1">
                {container.ticker} • {container.account} • {container.strategy}
                {container.uptime_seconds && ` • ${formatUptime(container.uptime_seconds)}`}
              </div>
            </div>

            <div className="flex items-center gap-2">
              {container.status === 'running' ? (
                <>
                  <button
                    onClick={() => handleAction('restart', container.name)}
                    disabled={loading === container.name}
                    className="btn btn-secondary p-2"
                    title="Restart"
                  >
                    <RotateCw size={16} />
                  </button>
                  <button
                    onClick={() => handleAction('stop', container.name)}
                    disabled={loading === container.name}
                    className="btn btn-danger p-2"
                    title="Stop"
                  >
                    <Square size={16} />
                  </button>
                </>
              ) : (
                <button
                  onClick={() => handleAction('start', container.name)}
                  disabled={loading === container.name}
                  className="btn btn-success p-2"
                  title="Start"
                >
                  <Play size={16} />
                </button>
              )}
            </div>
          </div>
        ))}

        {containers.length === 0 && (
          <div className="text-center text-dark-text-secondary py-8">
            No containers found
          </div>
        )}
      </div>
    </div>
  );
};
