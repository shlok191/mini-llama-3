import styled from 'styled-components';

import React, { useContext } from 'react';
import Records from '../contexts/records.tsx';
import { StyledSliderContainer, StyledSlider, StyledInputRange, StyledLabel } from './components.style.tsx';


interface SlidersProps {
    className?: string
}

const Sliders: React.FC<SlidersProps> = ({ className = '' }) => {
    const { temperature, top_k, setTemperature, setTopK } = useContext(Records);

    const handleTemperatureChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setTemperature(parseFloat(event.target.value));
    };

    const handleTopKChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setTopK(parseInt(event.target.value, 10));
    };

    return (
        <StyledSliderContainer className={className}>
            <StyledSlider>
                <StyledInputRange
                    id="temperatureSlider"
                    type="range"
                    min="0.05"
                    max="1"
                    step="0.05"
                    value={temperature}
                    onChange={handleTemperatureChange}
                    
                />
                <StyledLabel htmlFor="temperatureSlider">
                    Temperature: {temperature.toFixed(2)}
                </StyledLabel>
            </StyledSlider>
            <StyledSlider>
                <StyledInputRange
                    id="topKSlider"
                    type="range"
                    min="1"
                    max="96"
                    value={top_k}
                    onChange={handleTopKChange}
                />
                <StyledLabel htmlFor="topKSlider">
                    Top K Samples: {top_k}
                </StyledLabel>
            </StyledSlider>
        </StyledSliderContainer>
    );
};

export default Sliders;