import argparse
import json
import os

import openai
import time
from openai import OpenAI

NUM_SECONDS_TO_SLEEP = 0.5


def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4-0314',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': [{"type": "text",
                                 "text": content
                                 },
                                 {
                                 "type": "image_url",
                                 "image_url": {
                                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                                }}
                                ],
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']

def mixtral_get_eval(content: str, max_tokens: int):
    st = time.time()
    while True:
        try:
            api_key = "sk-PYxK2Ns557Zzc7NqiW41KXUekTiOUR61zUBBJf7lSzBSnpxR"
            base_url = "https://api.openai-proxy.org/v1"
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            chat_completion = client.chat.completions.create(
                messages=[{
                            'role': 'system',
                            'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                        }, {
                            'role': 'user',
                            'content': content,
                        }],
                model="gpt-4-turbo",
                max_tokens=max_tokens,
                temperature=0,
                top_p=0.1
            )   
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    print(f"{time.time()-st:.4f}s")
    return chat_completion.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    print(mixtral_get_eval("Are you gpt-4", 1000))
