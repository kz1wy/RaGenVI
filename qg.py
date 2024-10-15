import json
import torch
import argparse
import warnings

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")


class QuestionGenerator:
    def __init__(self, model_name, input_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.data = None
        with open(input_path, 'r', encoding='utf8') as f:
            self.data = json.load(f)

    def generate_question(self, max_length=128):
        """
            return a list of JSON object
            {
                id: int,
                context: String,
                questions: {
                    "beam_search": List of questions,
                    "sampling": List of questions
                }
            }
        """
        if not self.data:
            raise ValueError("Cannot load empty data.")

        generated_data = []
        for idx, item in enumerate(tqdm(self.data, desc="Generating Questions", unit="context")):
            context = item['context']
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Here we use Beam-search. It generates better quality queries, but with less diversity
                beam_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    length_penalty=2.0,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=5,
                    early_stopping=True
                )
                # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
                sampling_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    do_sample=True,
                    top_p=0.9,
                    top_k=20,
                    num_return_sequences=2,
                    early_stopping=True
                )

            beam_search_queries = [self.tokenizer.decode(beam_output, skip_special_tokens=True) for beam_output in
                                   beam_outputs]
            # we remove first output of sampling method, because it is potentially duplicated with beam output
            sampling_queries = [self.tokenizer.decode(sampling_output, skip_special_tokens=True) for sampling_output in
                                sampling_outputs]

            # Append the results to the generated_data list
            generated_data.append({
                "id": idx + 1,
                "context": context,
                "questions": {
                    "beam_search": beam_search_queries,
                    "sampling": sampling_queries
                }
            })

        return generated_data


def main(args):
    qg = QuestionGenerator(args.model_name, args.input_path)
    output_path = args.output_path
    context_question = qg.generate_question()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(context_question, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Question Generation")
    parser.add_argument("--model_name", default="doc2query/msmarco-vietnamese-mt5-base-v1", help="Model name")
    parser.add_argument("--input_path", default="data/raw/law_context_test.json", help="Input data path")
    parser.add_argument("--output_path", default="data/processed/context_question.json", help="Output data path")

    args = parser.parse_args()
    main(args)