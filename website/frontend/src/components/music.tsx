import React, { useEffect, useContext, useRef } from 'react';
import Records from '../contexts/records.tsx';

const Music: React.FC = () => {
    
    const { theme, music, setMusic } = useContext(Records);
    const audioRef = useRef<HTMLAudioElement>(null);
    useEffect(() => {
        
        // Sets the correct source file
        const src = theme ? 'pirate-music.mp3' : 'vanilla-music.mp3';
        
        if (audioRef.current) {
           audioRef.current.src = src;
        } 
        
        else {
            audioRef.current = new Audio(src);
            setMusic(audioRef.current);
        }

        audioRef.current.volume = 0.1;
        audioRef.current.loop = true;
        audioRef.current.play()
        .catch((error) => {
            console.error("Audio playback error:", error);
        });

        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
                audioRef.current.currentTime = 0;
              }
            }
         }, [theme, setMusic]);

       return (
        <audio ref={audioRef}  />
       );
};

export default Music;