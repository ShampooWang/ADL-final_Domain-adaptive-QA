"""TODO(squad_v2): Add a description here."""


import os
import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

ROOT_DIR = "/home/jgtf0322/Domain-Adaptive-QA/eqa_dataset"


# TODO(squad_v2): BibTeX citation
_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers
 to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but
 also determine when no answer is supported by the paragraph and abstain from answering.
"""


class SquadV2Config(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUADV2.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadV2Config, self).__init__(**kwargs)


class SquadV2(datasets.GeneratorBasedBuilder):
    """TODO(squad_v2): Short description of my dataset."""

    # TODO(squad_v2): Set up version.
    DATASET_LIST = ['boolq', 'complexq', 'complexwebq', 'comqa', 'drop', 'duorc_p', 'duorc_s',  'hotpotqa', 'newsqa', 'nq', 'quac', 'searchqa', 'squad', 'triviaqa', 'wikihop']
    BUILDER_CONFIGS = [SquadV2Config(name=tag, version=datasets.Version("2.0.0"), description="SQuAD plaint text version 2") for tag in DATASET_LIST]

    def _info(self):
        # TODO(squad_v2): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://rajpurkar.github.io/SQuAD-explorer/",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v2): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs

        train_path = os.path.join(ROOT_DIR, f'{self.config.name}/{self.config.name}_train.json')
        dev_path = os.path.join(ROOT_DIR, f'{self.config.name}/{self.config.name}_dev.json')

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path})
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(squad_v2): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "")
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"] for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }

