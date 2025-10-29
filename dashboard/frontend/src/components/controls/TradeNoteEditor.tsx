/**
 * Trade Note Editor - Inline editing for trade annotations
 */
import React, { useState } from 'react';
import { Edit2, Save, X, StickyNote } from 'lucide-react';
import { updateTradeNote } from '../../api/client';

interface TradeNoteEditorProps {
  tradeId: string;
  initialNote?: string | null;
  onSave?: (note: string) => void;
}

export const TradeNoteEditor: React.FC<TradeNoteEditorProps> = ({
  tradeId,
  initialNote,
  onSave,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [note, setNote] = useState(initialNote || '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSave = async () => {
    if (!note.trim()) {
      setIsEditing(false);
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await updateTradeNote(tradeId, note.trim());
      setIsEditing(false);
      onSave?.(note.trim());
      console.log('✓ Note saved');
    } catch (err: any) {
      console.error('Failed to save note:', err);
      setError('Failed to save note');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setNote(initialNote || '');
    setIsEditing(false);
    setError(null);
  };

  if (isEditing) {
    return (
      <div className="flex items-start gap-2">
        <div className="flex-1">
          <textarea
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Add note (e.g., 'Test new DCA levels', 'Market news impact')..."
            className="w-full px-3 py-2 bg-dark-bg border border-dark-border rounded-md text-sm text-dark-text-primary focus:outline-none focus:border-blue-500 resize-none"
            rows={2}
            maxLength={500}
            autoFocus
          />
          {error && (
            <p className="text-xs text-loss mt-1">{error}</p>
          )}
          <p className="text-xs text-dark-text-secondary mt-1">
            {note.length}/500 characters
          </p>
        </div>
        <div className="flex gap-1">
          <button
            onClick={handleSave}
            disabled={saving || !note.trim()}
            className="p-2 bg-profit hover:bg-profit-dark text-white rounded transition-colors disabled:opacity-50"
            title="Save note"
          >
            {saving ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              <Save size={14} />
            )}
          </button>
          <button
            onClick={handleCancel}
            disabled={saving}
            className="p-2 bg-dark-border hover:bg-dark-text-muted text-dark-text-primary rounded transition-colors"
            title="Cancel"
          >
            <X size={14} />
          </button>
        </div>
      </div>
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="group flex items-center gap-2 text-sm text-dark-text-secondary hover:text-dark-text-primary transition-colors"
    >
      {initialNote ? (
        <>
          <StickyNote size={14} className="text-blue-500" />
          <span className="text-dark-text-primary max-w-[200px] truncate">
            {initialNote}
          </span>
          <Edit2 size={12} className="opacity-0 group-hover:opacity-100 transition-opacity" />
        </>
      ) : (
        <>
          <Edit2 size={14} />
          <span>Add note...</span>
        </>
      )}
    </button>
  );
};
