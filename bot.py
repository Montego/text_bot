from nltk import edit_distance
import random

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Здравствуйте', 'Шалом'],
            'responses': ['Привет, юзер', 'Зравствуй']
        },
        'goodbye': {
            'examples': ['Пока', 'Всего доброго', 'До свидания'],
            'responses': ['Пока, юзер', 'Счастливо']
        },
        'thanks': {
            'examples': ['Спасибо', 'Спасибо большое!', 'Сенкс', 'Благодарю'],
            'responses': ['Вам спасибо']
        },
        'whatcanyoudo': {
            'examples': ['Что ты умеешь?', 'расскажи что умеешь'],
            'responses': ['Отвечать']
        },
        'name': {
            'examples': ['как тебя зовут?', 'кто ты?'],
            'responses': ['бот']
        },
        'weather': {
            'examples': ['Какая была погода?', 'Какая погода в Москве?'],
            'responses': ['Погода такая себе']
        },
    },
    'failure': [
        'Что это?',
        'Не понял вас',
        'Я на такое не умею отвечать'
    ]
}


def get_intent(text):
    intents = BOT_CONFIG['intents']
    for intent, value in intents.items():
        for example in value['examples']:
            dist = edit_distance(text, example)
            l = len(example)
            similarity = 1 - dist/l
            if similarity > 0.6:
                return intent


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_failure():
    failure_responses = BOT_CONFIG['failure']
    return random.choice(failure_responses)


def generate_answer(text):
    intent = get_intent(text)
    if intent:
        response = get_response_by_intent(intent)
        return response
    failure = get_failure()
    return failure


while True:
    text = input('Введите вопрос: ')
    answer = generate_answer(text)
    print(answer)
