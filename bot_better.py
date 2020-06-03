from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ДАнные
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

# размерность должна совпадать
X_text = []  # тексты           (4)    'Привет', 'Добрый день', 'Здравствуйте', 'Шалом'
y = []  # классы(интенты)       (4)   'hello', 'hello', 'hello', 'hello',

# Векторизация
for intent, value in BOT_CONFIG['intents'].items():
    for example in value['examples']:
        X_text.append(example)
        y.append(intent)


# Выбор векторайзера - влияет на score
# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X_text)
N = 100
scores = []
for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Выбор классификатора - влияет на score
    classificator = LogisticRegression()
    # classificator = SVC()
    classificator.fit(X_train, y_train)

    score = classificator.score(X_test, y_test)
    scores.append(score)

print(sum(scores) / N)
print(max(scores))
