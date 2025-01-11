import styled from 'styled-components';

export const MainContainer = styled.div`
    min-height: 90vh;
    width: 100%
    background-color: var(--card-background);
    display: flex;
    flex-direction: row;
    margin: 0;
    padding: 0;
`;

export const ContentContainer = styled.div`
    width: 70%;
    display: flex;
    flex-grow: 1;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    margin: 0;
`;

export const ContentCard = styled.div`
    background-color: var(--card-background);
    border-radius: 0.2rem;
    padding: 1.5rem;
    min-height: calc(100vh - 4rem);
    position: relative;
    transition: transform 0.2s ease-in-out, box-shadow 0.3s ease-in-out;
    width: 100%;
    box-shadow: 0px 5px 5px 5px rgba(0, 0, 0, 0.2);
    display: flex;
    flex-direction: column;
    z-index: 1;

    &:hover {
        transform: translateY(-1px); 
        box-shadow: 0px 8px 8px 8px rgba(0, 0, 0, 0.2);
`;

export const Title = styled.h1`
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-align: center;
    font-family: var(--title-font-family)
`;

export const ButtonContainer = styled.div`
    display: flex;
    justify-content: center;
    margin-bottom: -1rem;
    justify-content: flex-end;
    padding-right: 0.5rem;
`;

export const ThemeButton = styled.button`
  padding: 0.5rem 1rem;
  background-color: var(--button-color);
  color: var(--button-text-color);
  border-radius: 0.5rem;
  transition: background-color 0.3s ease-in-out;
    &:hover {
        background-color: var(--button-color);
    }
`;