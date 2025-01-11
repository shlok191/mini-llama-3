import React, { useState, useEffect, useContext } from 'react'
import Prompt from "../components/prompt.tsx"
import ToggleSwitch from "../components/toggle.tsx"
import GenerativeBox from "../components/generate.tsx"
import { LeftPillar, RightPillar } from "../components/pillar.tsx"
import { StyledTextDivider } from '../components/components.style.tsx'
import Music from '../components/music.tsx'
import SuggestionBubbles from '../components/suggestions.tsx'
import Sliders from "../components/sliders.tsx"
import Records from "../contexts/records.tsx"
import * as PageStyle from "./pages.style.tsx"

const WebPage: React.FC = () => {

    // Defining helpful state variables
    const [generatedText, setGeneratedText] = useState<string>('');
    const { theme, setTheme } = useContext(Records);

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
                        {theme ? "Ahoy, Matey! Spin a Yarn!" : "Let's Scribble a Story!"}
                    </PageStyle.Title>

                    {/* Prompt component */}
                    <SuggestionBubbles />
                    <Prompt />
                    <Sliders />
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: '900'}}>Generated Text</span>
                    </StyledTextDivider>
                    {/* GenerativeBox component */}
                    <GenerativeBox
                        model={theme ? 'blackbeard' : 'vanilla'}
                        onTextGenerated={handleTextGenerated}
                        typingSpeed={10}
                        fadeInDuration={50}
                    />
                    <StyledTextDivider>
                        <span style={{fontSize: '22px', fontWeight: '900'}}>Project Details</span>
                    </StyledTextDivider>
                </PageStyle.ContentCard>
            </PageStyle.ContentContainer>

            {/* Defining the right pillar */}
            <RightPillar />
        </PageStyle.MainContainer>
      );
    };

export default WebPage;