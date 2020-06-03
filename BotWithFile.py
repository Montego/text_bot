from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from nltk import edit_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
CLASSIFIER_THRESHOLD = 0.3
GENERATIVE_THRESHOLD = 0.7
dialogs = []
GENERATIVE_DIALOGS = []
X_text = []
y = []

with open('dialoges.txt') as f:
    data = f.read()

for dialog in data.split('\n\n'):
    replicas = []
    for replica in dialog.split('\n')[:2]:
        replica = replica[2:].lower()
        replicas.append(replica)
    if len(replicas) == 2 and 0 < len(replicas[0]) < 25 and 0 < len(replicas[1]) < 25:
        dialogs.append(replicas)

GENERATIVE_DIALOGS = dialogs[:100000]

for intent, value in BOT_CONFIG['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)

VECTORIZER = CountVectorizer()
X = VECTORIZER.fit_transform(X_text)

CLASSIFICATOR = LogisticRegression()
CLASSIFICATOR.fit(X, y)


def get_intent(text):
    probas = CLASSIFICATOR.predict_proba(VECTORIZER.transform([text]))
    max_proba = max(probas[0])
    if max_proba >= CLASSIFIER_THRESHOLD:
        index = list(probas[0]).index(max_proba)
        return CLASSIFICATOR.classes_[index]


def get_answer_by_generative_model(text):
    text = text.lower()
    # for question, answer in dialogs:
    for question, answer in GENERATIVE_DIALOGS:
        if abs(len(text) - len(question)) / len(question) < 1 - GENERATIVE_THRESHOLD:
            dist = edit_distance(text, question)
            l = len(question)
            similarity = 1 - dist / l
            if similarity > GENERATIVE_THRESHOLD:
                return answer


def get_response_by_intent(intent):
    responses = BOT_CONFIG['intents'][intent]['responses']
    return random.choice(responses)


def get_failure():
    failure_responses = BOT_CONFIG['failure']
    return random.choice(failure_responses)


stats = {
    'requests': 0,
    'by_script': 0,
    'by_generative_model': 0,
    'stubs': 0
}


def generate_answer(text):
    stats['requests'] += 1
    intent = get_intent(text)
    if intent:
        stats['by_script'] += 1
        response = get_response_by_intent(intent)
        return response
    answer = get_answer_by_generative_model(text)
    if answer:
        stats['by_generative_model'] += 1
        return answer
    stats['stubs'] += 1
    failure = get_failure()

    return failure


# while True:
#     text = input('Введите вопрос: ')
#     answer = generate_answer(text)
#     print(answer)


def start(update, context):
    update.message.reply_text('Hi!')


def help(update, context):
    update.message.reply_text('Help!')


def generate(update, context):

    answer = generate_answer(update.message.text)
    print(update.message.text, '--->', answer)
    print(stats)
    update.message.reply_text(answer)


def error(update, context):
    update.message.reply_text('Я работаю только с текстом')


def main():
    updater = Updater("TOKEN", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, generate))
    dp.add_error_handler(error)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
