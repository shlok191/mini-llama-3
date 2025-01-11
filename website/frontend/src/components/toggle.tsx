import React, { useContext } from 'react';
import { ToggleSwitchContainer, ToggleSwitchInput, ToggleSwitchSlider, ToggleLabel, ToggleSwitchBox } from './components.style.tsx';
import Records from '../contexts/records.tsx';

const ToggleSwitch: React.FC = () => {
    
    const { theme, setTheme } = useContext(Records);

    const handleToggle = () => {
        setTheme(prevTheme => !prevTheme);
    };

    return (
        <ToggleSwitchContainer>
            <ToggleLabel>
                {theme ? "Switch to Vanilla!" : "Arrr, Switch to Pirate!"}
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