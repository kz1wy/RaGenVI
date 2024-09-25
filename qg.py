import json
import hashlib
import torch
import argparse
import warnings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from collections import defaultdict

warnings.filterwarnings("ignore")

class QuestionGenerator:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.qg_pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_questions(self, data, batch_size, min_length, max_length):
        print("********** Question Generation Starts! **********")
        questions = []
        temp = []

        for d in data:
            title = d['title']
            for paragraph in d['paragraphs']:
                temp.append({"title": title, "context": paragraph['context']})

        for item in temp:
            context = item['context']
            input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)

            with torch.no_grad():
                beam_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=128,
                    length_penalty=2.0,
                    num_beams=5,
                    no_repeat_ngram_size=5,
                    num_return_sequences=1,
                    early_stopping=True
                )
                sampling_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=64,
                    do_sample=True,
                    top_p=0.95,
                    top_k=10,
                    num_return_sequences=1,
                    early_stopping=True
                )

            ques = [self.tokenizer.decode(out, skip_special_tokens=True) for out in sampling_outputs] + \
                   [self.tokenizer.decode(out, skip_special_tokens=True) for out in beam_outputs]
            questions.append(ques)

        return self.format_output(temp, questions)

    @staticmethod
    def load_data(path):
        with open(path, 'r', encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_data(data, path):
        with open(path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    @staticmethod
    def format_output(temp, questions):
        cqa = {"data": []}
        title2ctx = defaultdict(list)
        ctx2qas = defaultdict(list)

        for item, ques in zip(temp, questions):
            title, context = item['title'], item['context']
            title2ctx[title].append(context)

            for q in ques:
                str2hash = title + context + q
                qid = hashlib.md5(str2hash.encode()).hexdigest()
                ctx2qas[context].append({
                    "id": qid,
                    "question": q + "?",
                })

        for title, contexts in title2ctx.items():
            cqa_index = {
                "title": title,
                "paragraphs": []
            }
            for c in contexts:
                paragraph = {
                    "context": c,
                    "qas": ctx2qas[c]
                }
                if paragraph not in cqa_index['paragraphs']:
                    cqa_index["paragraphs"].append(paragraph)
            cqa['data'].append(cqa_index)

        print("********** Question Generation Ends! **********")
        return cqa


def main(args):
    qg = QuestionGenerator(args.model_name)
    df = qg.load_data(args.input_path)
    cqa = qg.generate_questions(df['data'], args.batch_size, args.min_length, args.max_length)
    qg.save_data(cqa, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Question Generation")
    parser.add_argument("--model_name", default="doc2query/msmarco-vietnamese-mt5-base-v1", help="Model name")
    parser.add_argument("--input_path", default="data/raw/commercial_law_test.json", help="Input data path")
    parser.add_argument("--output_path", default="data/processed/commercial_question.json", help="Output data path")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--min_length", type=int, default=64, help="Minimum length")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length")

    args = parser.parse_args()
    main(args)