
import React from 'react';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import { X, Video } from 'lucide-react';

export const VideoClipPlayerModal: React.FC = () => {
  const { activeEvidenceClip, closeEvidenceClip } = useInvestigationStore();

  if (!activeEvidenceClip) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(20,17,13,0.7)' }}
    >
      <div
        className="folder-pull"
        style={{
          width: '100%',
          maxWidth: '800px',
          background: '#E8DCC0', // manila
          border: '1px solid rgba(176,141,87,0.3)',
          boxShadow: '0 12px 40px rgba(20,17,13,0.4)',
          position: 'relative',
        }}
      >
        {/* Brass pin top center */}
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 12,
            height: 12,
            borderRadius: '50%',
            background: `radial-gradient(circle at 38% 35%, #d4aa6e, #B08D57 60%, #7a5c2e)`,
            boxShadow: '0 1px 3px rgba(20,17,13,0.35)',
            zIndex: 1,
          }}
        />

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '24px 20px 14px 20px',
            borderBottom: `1px solid rgba(176,141,87,0.35)`,
            marginTop: 8,
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              fontFamily: "'Source Serif 4', serif",
              fontSize: 16,
              fontWeight: 700,
              color: '#1A1512',
            }}
          >
            <Video style={{ width: 18, height: 18, color: '#B08D57' }} />
            <span>Forensic Evidence: {activeEvidenceClip.title}</span>
          </div>
          <button
            onClick={closeEvidenceClip}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#6B5545',
            }}
          >
            <X style={{ width: 16, height: 16 }} />
          </button>
        </div>

        <div style={{ padding: 20 }}>
          <div style={{ background: '#1A1512', padding: 4, borderRadius: 2, border: '1px solid rgba(26,21,18,0.2)' }}>
            <video 
              src={activeEvidenceClip.url} 
              controls 
              autoPlay 
              className="w-full h-full object-contain aspect-video"
            />
          </div>
        </div>

        <div
          style={{
            padding: '12px 20px',
            background: '#F0E8D0',
            borderTop: `1px solid rgba(176,141,87,0.35)`,
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 10,
            color: '#6B5545',
            textTransform: 'uppercase',
            letterSpacing: '0.05em'
          }}
        >
          SOURCE: {activeEvidenceClip.url}
        </div>
      </div>
    </div>
  );
};
