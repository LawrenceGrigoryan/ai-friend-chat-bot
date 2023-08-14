"""
Data preparation scripts for fine-tuning
"""
import logging
from typing import Union, List, NoReturn
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets

from prompt_templates import (
    USER_PROMPT,
    MODEL_OUTPUT,
    EMPATHETIC_SYSTEM_PROMPT,
    DAILY_SYSTEM_PROMPT
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def concat_dialogue(messages: Union[List[str], pd.Series],
                    system_prompt: str) -> str:
    """
    Concatenate dialogue sentences into one
    dialogue sample using LLaMa2 training data format

    Args:
        messages (Union[List[str], pd.Series]): dialogue messages
        system_prompt (str): System prompt to add to the dialogue

    Returns:
        str: concatenated dialogue string
    """
    if type(messages) is not list:
        messages = messages.tolist()

    for i in range(len(messages)):
        if i % 2 == 0:
            messages[i] = USER_PROMPT.format(user_message=messages[i].strip())
        else:
            messages[i] = MODEL_OUTPUT.format(model_output=messages[i].strip())

    return system_prompt + ' '.join(messages)


def prepare_empathetic_dataset(dataset: Dataset) -> Dataset:
    """
    Prepare the given empathetic dataset into suitable LLM format

    Args:
        dataset (Dataset): dataset

    Returns:
        Dataset: prepared dataset
    """
    dataset.set_format('pandas')
    df = dataset[:]
    # Filter out the most negative sentiments
    negative_contexts = ['angry', 'jealous', 'disgusted', 'annoyed',
                         'anxious', 'devastated', 'terrified', 'furious']
    df = df[~df.context.isin(negative_contexts)]
    # Filter out conversations with less than 1 sentence (they are buggy and useless)
    conv_sizes = df.groupby(['conv_id']).size()
    convs2keep = conv_sizes[conv_sizes > 1].index
    df = df[df.conv_id.isin(convs2keep)]
    # Concat dialogues
    df_grouped = df.groupby(['conv_id']).utterance \
                   .apply(
        lambda x: concat_dialogue(x, EMPATHETIC_SYSTEM_PROMPT)
        ).reset_index()
    df_grouped = df_grouped.rename(columns={'utterance': 'sample'})
    df_grouped['sample'] = df_grouped['sample'].str.replace('_comma_', ', ')
    df_grouped = df_grouped[['sample']]
    dataset_prepared = Dataset.from_pandas(df_grouped)

    return dataset_prepared


def prepare_daily_dataset(dataset: Dataset) -> Dataset:
    """
    Prepare the given empathetic dataset into suitable LLM format

    Args:
        dataset (Dataset): dataset

    Returns:
        Dataset: prepared dataset
    """
    dataset.set_format('pandas')
    df = dataset[:]
    df_grouped = df.dialog.apply(
        lambda x: concat_dialogue(x, DAILY_SYSTEM_PROMPT)
        )
    df_grouped = df_grouped.reset_index() \
                           .drop(['index'], axis=1) \
                           .rename(columns={'dialog': 'sample'})
    dataset_prepared = Dataset.from_pandas(df_grouped)

    return dataset_prepared


def prepare_data() -> NoReturn:
    """
    Loads datasets, concatenates them and saves separate splits

    Returns:
        NoReturn
    """
    data_dir = Path('../data/')
    dataset_name1 = 'empathetic_dialogues'
    dataset_name2 = 'daily_dialog'

    splits = ['train', 'validation', 'test']
    for split in splits:
        logging.info('Preparing "{split}" split')
        logging.info('Preparing empathetic dialogues')
        emp_dataset = load_dataset(dataset_name1, split=split)
        emp_dataset_prep = prepare_empathetic_dataset(emp_dataset)
        logging.info('Preparing daily dialogues')
        daily_dataset = load_dataset(dataset_name2, split=split)
        daily_dataset_prep = prepare_daily_dataset(daily_dataset)
        logging.info('Concatenating datasets and saving current split to disk')
        split_concat = concatenate_datasets([emp_dataset_prep,
                                             daily_dataset_prep])
        split_concat.save_to_disk(data_dir.joinpath(f'{split}.hf'))


if __name__ == '__main__':
    prepare_data()
