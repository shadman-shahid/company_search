
# Running the `train.py` file.

## The Application

> This application is used for training a Named Entity Recognition (NER) model using the Transformers library.

## Library Dependencies

1. **NLTK** 
2. **torch**
3. **Pandas**
4. **json:** A module in Python that provides a method for encoding and decoding JSON objects.
5. **sklearn**

### Important Variables:

1. `model_checkpoint`: This is the model checkpoint used as the basis for fine-tuning.

2. `tokenizer`: This is the tokenizer corresponding to the model checkpoint.

3. `config`: This is the configuration object corresponding to the model checkpoint.

4. `model`: This is the model used for fine-tuning.

5. `device`: This is the device (CPU or GPU) used for training.

6. `training_args`: These are the arguments used for training.

7. `trainer`: This is the trainer used for fine-tuning the model.

### Functions:

1. `compute_metrics(eval_pred: EvalPrediction)`: This function calculates the precision, recall, and F1 score based on the predicted and true labels.

2. `get_labels(text, labels)`: This function maps entities to tokens in a given sentence.

3. `convert_data_to_ner_format(data)`: This function converts the data into a format suitable for Named Entity Recognition (NER).

### Running The Application

The script loads a dataset from a JSON lines file, converts the data into the format required for Named Entity Recognition (NER), tokenizes the data, and fine-tunes a pre-trained model on this data.

The training script fine-tunes the model for 10 epochs, logging the training loss every 200 steps. The best model is saved based on the evaluation loss.

## Note

1. Ensure the jsonl file `'./data/final_dataset.jsonl'` exists and contains the correct data in a suitable format for Named Entity Recognition. The `final_dataset.jsonl` file is inferred to contain JSON objects, one per line, where each JSON object represents an item in the dataset. 
> The file must have the following structure, or must be converted to the following structure for the code to work. **Note**: This is the standard structure for dataset prepared through **doccano** package, that we have used.
```json
{
    "text": "Amazon is an e-commerce platform. But it also has online streaming services. ",
    "entities": [
        {
            "start_offset": 14,
            "end_offset": 23,
            "label": "PROD"
        },
        ...
    ]
}
```
> This `.jsonl` file can be processed line by line, with each line being a separate JSON object representing a single data point. Each line's JSON object has a `text` field that is a string of text, and an `entities` field that is a list of dictionaries, each containing `start_offset`, `end_offset`, and `label` fields.

2. The `torch.device()` function is used to move the model and data to a GPU, if available. Ensure that your environment has a compatible GPU if you want to use this feature.

3. The model_checkpoint in the script is "distilbert-base-uncased". You can replace this with the path to a different model checkpoint if desired.

4. The training_args object controls the parameters for training. You can customize these parameters as needed.

5. The script uses a basic split of 80-20 for the training and testing datasets.

6. This script trains a NER model. Make sure that the data is labeled in a suitable format for Named Entity Recognition.
