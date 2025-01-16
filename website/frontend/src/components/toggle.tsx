import React, { useContext } from 'react';
import { ToggleSwitchContainer, ToggleSwitchInput, ToggleSwitchSlider, ToggleLabel, ToggleSwitchBox } from './components.style.tsx';
import Records from '../contexts/records.tsx';

const ToggleSwitch: React.FC = () => {
    
    const { theme, setTheme, setGenerate, setPrompt } = useContext(Records);

    const handleToggle = () => {
        
        setTheme(prevTheme => !prevTheme);
        setPrompt('');
    };

    return (
        <ToggleSwitchContainer>
            <ToggleLabel>
                {theme ? "Switch to Vanilla!" : "Arrr, Switch to Pirate!"}
                <span style={{
                   display: 'block',
                   fontSize: '1.0rem',
                   marginTop: '0.2rem',
                   color: 'var(--common-text-color)'
                  }}>
                    (Different Model)
                  </span>
            </ToggleLabel>
            
            <ToggleSwitchBox>
                <ToggleSwitchInput
                    type="checkbox"
                    checked={theme}
                />
                <ToggleSwitchSlider onClick={handleToggle}/>
            </ToggleSwitchBox>
        </ToggleSwitchContainer>
    );
};

export default ToggleSwitch;