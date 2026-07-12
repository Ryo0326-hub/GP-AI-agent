import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  Sequence,
  staticFile,
  useCurrentFrame,
} from 'remotion';
import type {FC} from 'react';

type Slide = {
  image: string;
  audio: string;
  duration: number;
  caption: string[];
  label: string;
};

const slides: Slide[] = [
  {
    image: 'slides/slide-01.png',
    audio: 'audio/slide-01.mp3',
    duration: 1095,
    label: 'GP-AI-Agent',
    caption: [
      'GP-AI-Agent is a local-first AI agent for the AMD Hackathon.',
      'It decides whether a task can be answered safely on the local machine.',
      'Difficult or untrusted tasks are escalated to a stronger Fireworks model.',
      'The goal is simple: spend tokens only when they create real value.',
    ],
  },
  {
    image: 'slides/slide-02.png',
    audio: 'audio/slide-02.mp3',
    duration: 945,
    label: 'The problem',
    caption: [
      'Using a strong model for every prompt can be costly and slow.',
      'Simple questions should not automatically take the same paid path as complex code tasks.',
      'Track 1 also imposes real limits: 4 GB RAM, 2 vCPU, and a 10-minute runtime.',
      'The challenge is to stay reliable inside those limits.',
    ],
  },
  {
    image: 'slides/slide-03.png',
    audio: 'audio/slide-03.mp3',
    duration: 975,
    label: 'The solution',
    caption: [
      'A learned router makes a small local prediction before any external call.',
      'It classifies the task and estimates whether local work should escalate.',
      'The agent verifies local work whenever possible.',
      'This routing decision is fast and uses zero Fireworks tokens.',
    ],
  },
  {
    image: 'slides/slide-04.png',
    audio: 'audio/slide-04.mp3',
    duration: 1260,
    label: 'Technical architecture',
    caption: [
      'The router is a compact logistic classifier trained from 360 measured local outcomes.',
      'The local solver is Qwen 2.5 1.5B in a compressed Q4 format through llama.cpp.',
      'Safety checks recompute arithmetic, test code, and validate selected outputs.',
      'If the local result is not trustworthy, the system escalates instead of guessing.',
    ],
  },
  {
    image: 'slides/slide-05.png',
    audio: 'audio/slide-05.mp3',
    duration: 1500,
    label: 'Measured results',
    caption: [
      'The earlier API-first approach used 9,685 API tokens.',
      'The v11 router profile estimates 1,824 API tokens with 97.14 percent projected accuracy.',
      'The 19-task rehearsal completed every task in 70.5 seconds.',
      'The 80-task pass finished in 207.4 seconds with zero fallbacks and 130 passing tests.',
    ],
  },
  {
    image: 'slides/slide-06.png',
    audio: 'audio/slide-06.mp3',
    duration: 945,
    label: 'Why it matters',
    caption: [
      'GP-AI-Agent is designed to be efficient and dependable at the same time.',
      'Local-first does not mean local-only.',
      'Every task gets the cheapest reliable path, with a stronger model ready when needed.',
      'The live walkthrough shows the routing decision, answer path, and token comparison.',
    ],
  },
];

export const totalDurationInFrames = slides.reduce((total, slide) => total + slide.duration, 0);

const SlideScene: FC<{slide: Slide}> = ({slide}) => {
  const frame = useCurrentFrame();
  const captionWindow = Math.ceil(slide.duration / slide.caption.length);
  const activeCaption = Math.min(slide.caption.length - 1, Math.floor(frame / captionWindow));
  const captionProgress = frame % captionWindow;
  const captionOpacity = interpolate(captionProgress, [0, 12, captionWindow - 12, captionWindow], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const scale = interpolate(frame, [0, slide.duration], [1, 1.035], {extrapolateRight: 'clamp'});
  const fade = interpolate(frame, [0, 12, slide.duration - 14, slide.duration], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <AbsoluteFill style={{backgroundColor: '#09110d', opacity: fade, overflow: 'hidden'}}>
      <Audio src={staticFile(slide.audio)} />
      <Img
        src={staticFile(slide.image)}
        style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${scale})`}}
      />
      <div style={{position: 'absolute', top: 38, right: 56, color: '#b6f64b', fontFamily: 'Arial, sans-serif', fontSize: 25, fontWeight: 700, letterSpacing: 1.5}}>
        {slide.label.toUpperCase()}
      </div>
      <div style={{position: 'absolute', left: 72, right: 72, bottom: 58, minHeight: 108, borderRadius: 20, background: 'rgba(9, 17, 13, 0.86)', border: '1px solid rgba(182,246,75,0.35)', display: 'flex', alignItems: 'center', padding: '20px 30px', boxSizing: 'border-box'}}>
        <div style={{opacity: captionOpacity, color: '#f5f7f4', fontFamily: 'Arial, sans-serif', fontSize: 37, fontWeight: 600, lineHeight: 1.2}}>
          {slide.caption[activeCaption]}
        </div>
      </div>
      <div style={{position: 'absolute', right: 76, bottom: 185, color: '#62e29a', fontFamily: 'Arial, sans-serif', fontSize: 19, fontWeight: 700}}>
        {activeCaption + 1} / {slide.caption.length}
      </div>
    </AbsoluteFill>
  );
};

export const HackathonPitch: FC = () => {
  let start = 0;
  return (
    <AbsoluteFill style={{backgroundColor: '#09110d'}}>
      {slides.map((slide) => {
        const from = start;
        start += slide.duration;
        return (
          <Sequence key={slide.image} from={from} durationInFrames={slide.duration}>
            <SlideScene slide={slide} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
