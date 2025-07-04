import React from 'react';

interface HeaderProps {
  onCompare?: () => void;
  showCompareButton?: boolean;
  onAdminTraining?: () => void;
  showAdminButton?: boolean;
}

const Header: React.FC<HeaderProps> = ({ onCompare, showCompareButton = false, onAdminTraining, showAdminButton = false }) => {
  return (
    <header className="bg-wrestling-charcoal border-b-4 border-wrestling-red p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center">
          <div className="flex-1"></div>
          
          <div className="text-center">
            <h1 className="text-5xl font-black tracking-wider uppercase text-white">
              HEATMETER
            </h1>
            <p className="text-wrestling-gray mt-2 tracking-wider">
              Live sentiment from the wrestling community
            </p>
          </div>
          
          <div className="flex-1 flex justify-end gap-2">
            {showCompareButton && onCompare && (
              <button
                onClick={onCompare}
                className="wrestling-button text-sm px-4 py-2"
              >
                COMPARE
              </button>
            )}
            {showAdminButton && onAdminTraining && (
              <button
                onClick={onAdminTraining}
                className="wrestling-button text-sm px-4 py-2"
                style={{backgroundColor: '#6b7280'}}
              >
                🔧 ADMIN
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;