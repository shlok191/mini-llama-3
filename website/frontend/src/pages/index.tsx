import React, { useState, useEffect, useContext } from 'react'

import Records from "../contexts/records.tsx"

import Prompt from "../components/prompt.tsx"
import ToggleSwitch from "../components/toggle.tsx"
import GenerativeBox from "../components/generate.tsx"

import { LeftPillar, RightPillar } from "../components/pillar.tsx"
import { StyledTextDivider } from '../components/components.style.tsx'

import Header from '../components/descriptions/header.tsx'
import CurrentDetails from '../components/descriptions/cuda.tsx'
import TrainingDetails from '../components/descriptions/training.tsx'
import Links from '../components/descriptions/links.tsx'
import Credits from '../components/descriptions/credits.tsx'

import SuggestionBubbles from '../components/suggestions.tsx'
import Sliders from "../components/sliders.tsx"
import * as PageStyle from "./pages.style.tsx"

const WebPage: React.FC = () => {

    // Defining helpful state variables
    const [generatedText, setGeneratedText] = useState<string>('');
    const { theme } = useContext(Records);

    // Callback to handle text generation completion
    const handleTextGenerated = (text: string) => {
        setGeneratedText(text);
    };

    // Defining the CSS theme that is to be used!
    useEffect(() => {
        document.documentElement.className = theme ? 'pirate-ui' : 'vanilla-ui';
    }, [theme]);

    return (
        <PageStyle.MainContainer>
            
            {/* TODO: Adding background music! <Music />*/}
            
            {/* Defining the left pillar */}   
            <LeftPillar />

            {/* Main content card */}
            <PageStyle.ContentContainer>
                <PageStyle.ContentCard>
                    
                    {/* Theme toggle button */}
                    <PageStyle.ButtonContainer>
                        <ToggleSwitch />
                    </PageStyle.ButtonContainer>

                    {/* The title tag */}
                    <PageStyle.Title>
                        {theme ? "Ahoy, Matey! Spin a Yarn!" : "Let's Write a Story!"}
                    </PageStyle.Title>

                    {/* Prompt component */}
                    <SuggestionBubbles />
                    
                    <Prompt />
                    <Sliders />
                    
                    {/* GenerativeBox component */}
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: 'bolder'}}>Generated Text</span>
                    </StyledTextDivider>
                    
                    <GenerativeBox
                        model={theme ? 'blackbeard' : 'vanilla'}
                        onTextGenerated={handleTextGenerated}
                        typingSpeed={10}
                        fadeInDuration={50}
                    />

                    {/* Project Description */}
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: 'bolder'}}>Project Description</span>
                    </StyledTextDivider>
                    <Header />

                    {/* Model Details */}
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: '900'}}>Model Implementation</span>
                    </StyledTextDivider>
                    
                    <CurrentDetails />
                    
                    {/* Training Details */}
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: '900'}}>Training Implementation</span>
                    </StyledTextDivider>

                    <TrainingDetails />

                    {/* Credits & Links */}
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: '900'}}>Credits & Links</span>
                    </StyledTextDivider>
                    
                    <Credits />
                    <Links />

                </PageStyle.ContentCard>
            </PageStyle.ContentContainer>

            {/* Defining the right pillar */}
            <RightPillar />
        </PageStyle.MainContainer>
      );
    };

export default WebPage;