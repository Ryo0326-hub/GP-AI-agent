import {Composition} from 'remotion';
import {HackathonPitch, totalDurationInFrames} from './video';

export const RemotionRoot = () => {
  return (
    <Composition
      id="HackathonPitch"
      component={HackathonPitch}
      durationInFrames={totalDurationInFrames}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
