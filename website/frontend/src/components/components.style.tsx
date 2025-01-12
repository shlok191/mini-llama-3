import styled from 'styled-components';

export const StyledPrompt = styled.div`
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
`;

export const StyledInput = styled.input`
    padding: 10px;
    margin-right: 10px;
    border: 1px solid var(--common-text-color);
    border-radius: var(--common-border-radius);
    flex: 1;
    margin-right: 1rem;
    background-color: var(--prompt-box-color);
    outline: none;
    color: var(--button-text-color);
    font-family: var(--font-family);
`;

export const StyledTextDivider = styled.div`
    display: flex;
    align-items: center;
    text-align: center;
    color: rgba(0, 0, 0, 1.);
    margin: 10px 0; /* Adjust margin as needed */

    &:before, &:after {
        content: '';
        flex: 1;
        border-bottom: 1px solid rgba(0, 0, 0, 0.9);
        box-shadow: 0px 1px 1px 0.5px rgba(0, 0, 0, 0.1);
    }

    &:before {
        margin-right: 1em;
        margin-left: 10em;
    }

    &:after {
        margin-left: 1em;
        margin-right: 10em;
    }
`;

export const StyledButton = styled.button`
    padding: 10px 15px;
    background-color: var(--button-color);
    color: var(--button-text-color);
    border: 1px solid var(--common-text-color);
    border-radius: var(--common-border-radius);
    cursor: pointer;
    white-space: nowrap;
    font-family: var(--font-family);
`;

export const ToggleSwitchContainer = styled.div`
    position: relative;
    display: flex;
    align-items: center;
    width: 238px;
    height: 30px;
`;

export const ToggleSwitchBox = styled.div`
    position: relative;
    display: inline-block;
    width: 50px;
    height: 20px;
`;

export const ToggleLabel = styled.span`
    margin-right: 25px;
    color: #black;
    font-size: 20px;
`

export const ToggleSwitchInput = styled.input`
    opacity: 0;
    width: 0;
    height: 0;

    &:checked + span {
        background-color: var(--primary-color);
    }
    
    &:checked + span:before {
        transform: translateX(30px);
    }
`;

export const ToggleSwitchSlider = styled.span`
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: .4s;
    border-radius: 20px;
    
    &:before {
        position: absolute;
        content: "";
        height: 16px;
        width: 16px;
        left: 2px;
        bottom: 2px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
`;

export const Pillar = styled.div`
    width: 15%;
    height: 120vh;
    background-color: var(--pillar-background);
    padding: 0.2rem;
    top: 0;
    position: sticky;
    display: flex;
    flex-direction: column; 
    justify-content: flex-start;
    transition: box-shadow 0.2s ease-in-out;
    z-index: 0;

    &:hover {
        box-shadow: 0px 5px 5px 5px rgba(0, 0, 0, 0.1); // Shadow to the right when hovered
        z-index: 2;
    }
`;

export const Image = styled.img<{ theme }>`
  position: absolute;
  object-fit: contain
  height: auto;
  transition: opacity 0.5s ease-in-out;
`;

export const StyledSliderContainer = styled.div`
    display: flex;
    align-items: center;
    space-x: 4;
    width: 100%;
    margin-bottom: 20px;
`;

export const StyledSlider = styled.div`
    flex: 0.2;
`;

export const StyledInputRange = styled.input`
    width: 75%;
    height: 0.3rem;
    background-color: var(--common-text-color);
    border-radius: 0.5rem;
    -webkit-appearance: none;
    cursor: pointer;
    
    &::-moz-range-thumb {
      width: 0.75rem;
      height: 0.75rem;
      background: var(--prompt-box-color);
      cursor: pointer;
      border-radius: 50%;
    }
`;

export const StyledLabel = styled.label`
    display: block;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--common-text-color);
    padding-left: 20px;
`;

export const StyledSuggestionContainer = styled.div`
    display: flex;
    justify-content: center;
    margin-bottom: 0.5rem;
    width: 100%;
    overflow-x: auto;
`;

export const StyledSuggestionBubble = styled.div`
    background-color: var(--secondary-color);
    color: var(--common-text-color);
    border: 1px solid var(--common-text-color);
    padding: 0.5rem 0.8rem;
    border-radius: 1rem;
    margin-right: 0.5rem;
    cursor: pointer;
    white-space: nowrap;
    transition: background-color 0.3s ease;
    font-size: 15px;

    &:hover {
        background-color: var(--primary-color);
    }
`;