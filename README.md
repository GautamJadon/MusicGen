# Music Generation System using Deep Learning

This project involves generating music using deep learning techniques, particularly a Bidirectional LSTM model enhanced with an attention mechanism. The model is trained on a dataset of MIDI files to generate new music sequences based on input parameters such as musical key, time signature, and mood.

## Features
- **MIDI Data Processing**: Preprocesses MIDI files to extract musical notes and chords.
- **Model Architecture**: Utilizes a Bidirectional LSTM model with an attention mechanism for music sequence prediction.
- **Humanization**: Introduces randomness to rhythm and volume for more natural-sounding music generation.
- **Customizable Generation**: Allows users to specify key, time signature, and mood preferences before generating music.
- **Music Output**: The generated music is saved as a MIDI file and can be played back.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/music-generation-system.git
    cd music-generation-system
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The model is trained on a collection of MIDI files. The dataset should be placed in the `Data/archive` folder. The system will automatically scan this directory for `.mid` and `.midi` files.

## Usage

To run the music generation system:

1. Ensure your MIDI dataset is placed in the `Data/archive` directory.
2. Run the following command:
    ```bash
    python music_generation_system.py
    ```

3. The program will:
   - Load and preprocess MIDI files.
   - Train the model or load an existing one from the `Model/music_generation_model.h5` path.
   - Allow you to input musical parameters (key, time signature, mood).
   - Generate a new music sequence and save it as `generated_music.mid`.
   - Play the generated music.

## Model

The model uses a **Bidirectional LSTM** architecture with an **Attention Mechanism**:
- **Bidirectional LSTM** layers capture sequential dependencies from both directions.
- **Attention Layer** helps the model focus on important parts of the input sequence during generation.

### Model Training
- The model is trained using the sequences of musical notes and chords extracted from the MIDI files.
- If the model doesn't already exist, it will be trained from scratch on the dataset.
- The trained model is saved as `Model/music_generation_model.h5` for future use.

## Files

- `music_generation_system.py`: The main script to run the music generation system.
- `Model/music_generation_model.h5`: The saved trained model (if it exists).
- `Data/archive`: Folder where your MIDI files are stored.
- `generated_music.mid`: The output file generated by the system.

## Requirements

- Python 3.x
- TensorFlow
- Music21
- Numpy
- Other dependencies are listed in the `requirements.txt`.

## Contributing

Feel free to fork this project and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
