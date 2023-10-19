

# Tuning Whisper with Cog

This is a repository for tuning Whisper using Cog. It's beta-ish at the moment, I'd say. 

## Preparing your data 

The easiest way to do this is to put all of your audio into one folder, and transcriptions into a separate folder. For every audio file named "foo.mp3", you should have a separate transcription called "foo.txt". 

Once this is set up, run `python process_data.py --audio_files path/to/audio --transcription_files path/to/transcripts`. This should spit out a `whisper_dataset.tar.gz` file. Great! You've prepped your data. 

## Training Whisper

You can see the various arguments to train whisper in `train.py`. Definitely recommend having an evaluation dataset to go along with your training dataset (prepared in the same way as above). 

To train Whisper with that dataset, use the Replciate Training API to run a training on the most recent version of [`daanelson/whisper-tune`](https://replicate.com/daanelson/whisper-tune). Make sure to upload your training & validation files to a publicly accessible bucket somewhere. 

You'll also want to select the model to train; I'd recommend large-v2, though if you want to overfit against tiny or small to validate your dataset first, that's valid too. 

You can monitor the performance of your training in the logs in the Training on Replicate. 

## Whisper inference

Right now, post training you'll have a fine-tuned Whisper model on Replicate that will use the huggingface API to do transcription. Serving the trained whisper in a faster serving framework (i.e. whisperX) is a TODO.

