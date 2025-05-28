from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
import logging
from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter

# Здесь реализована логика чат бота с несколькими pdf файлами

# Файлы PDF с рекомендациями
pdf_files = {
    'КР359_3.pdf': 'J45 Бронхиальная астма',
    'КР708_2.pdf': 'К29 Гастрит и дуоденит у взрослых',
    'КР837_1.pdf': 'K29.0 Гастрит и дуоденит у детей'
}

# Создание отдельных индексов для каждого документа
vector_stores = {}


def create_vectorstore_for_file(file_path):
    #Создает векторный индекс для отдельного файла
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load_and_split(text_splitter)

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    vector_store = FAISS.from_documents(documents, embedding=embedding)
    return vector_store


# Создать индексы для каждого файла
for file, title in pdf_files.items():
    vector_stores[title] = create_vectorstore_for_file(file)

# Бот и диспетчер
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="7642713807:AAG-7hYwDH6qACk_yHr_YeCFQMmwvpM6IcA")
# Диспетчер
dp = Dispatcher()


# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="Диабет"),
            types.KeyboardButton(text="Гастрит"),
            types.KeyboardButton(text="Астма"),
            types.KeyboardButton(text="Глаукома")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True, input_field_placeholder="Выберите диагноз")
    await message.answer("Выберите диагноз", reply_markup=keyboard)


# Обработчик выбора диагноза "Гастрит"
@dp.message(F.text.lower() == "гастрит")
async def gastritis_handler(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="K29.0 Гастрит и дуоденит у детей"),
            types.KeyboardButton(text="К29 Гастрит и дуоденит у взрослых")
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите категорию"
    )
    await message.answer("Выберите категорию гастрита:", reply_markup=keyboard)

@dp.message(F.text.lower() == "к29 гастрит и дуоденит у взрослых")
async def gastritis_adults_handler(message: types.Message):
    await message.reply("Введите ваш вопрос:")

@dp.message(F.text.lower() == "к29.0 гастрит и дуоденит у детей")
async def gastritis_child_handler(message: types.Message):
    await message.reply("Введите ваш вопрос:")


@dp.message(F.text.lower() == "астма")
async def gastritis_handler(message: types.Message):
    kb = [
        [
            types.KeyboardButton(text="J45 Бронхиальная астма"),

        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберите категорию"
    )
    await message.answer("Выберите категорию астмы:", reply_markup=keyboard)

@dp.message(F.text.lower() == "j45 бронхиальная астма")
async def astma_adults_handler(message: types.Message):
    await message.reply("Введите ваш вопрос:")

# Обработка вопросов пользователя
@dp.message()
async def process_user_query(message: types.Message):
    question = message.text.strip()

    # Если пользователь хочет выйти
    if question.lower() == 'exit':
        await message.reply("До свидания!")
        return

    # Используем соответствующий векторный индекс
    relevant_title = "К29 Гастрит и дуоденит у взрослых"
    vector_store = vector_stores.get(relevant_title)

    relevant_title = "К29.0 Гастрит и дуоденит у детей"
    vector_store = vector_stores.get(relevant_title)

    relevant_title = "J45 Бронхиальная астма"
    vector_store = vector_stores.get(relevant_title)

    if not vector_store:
        await message.reply("Ошибка: документ не загружен.")
        return

    # Получаем релевантные фрагменты текста
    results = vector_store.similarity_search(question)

    if results:
        response_message = "\n\n".join([doc.page_content for doc in results])
        await message.reply(f"Результаты поиска:\n{response_message}")
    else:
        await message.reply("Ответ не найден.")


# Запуск приложения
async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())































# Список путей к файлам
#pdf_files = ['КР359_3.pdf', 'КР708_2.pdf', 'КР792_1.pdf']

# Создаем пустой список для хранения документов
#all_documents = []

# Текст-сплиттер
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
#                                              chunk_overlap=100)

#for file in pdf_files:
    # Загружаем каждый файл и делим на части
#    loader = UnstructuredPDFLoader(file)
#    documents = loader.load_and_split(text_splitter)

    # Добавляем разделенные документы в общий список
#    all_documents.extend(documents)

# Теперь splitted_data содержит все страницы всех файлов
#splitted_data = all_documents

#model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#model_kwargs = {'device': 'cpu'}
#encode_kwargs = {'normalize_embeddings': False}
#embedding = HuggingFaceEmbeddings(model_name=model_name,
#                                  model_kwargs=model_kwargs,
#                                  encode_kwargs=encode_kwargs)

#vector_store = FAISS.from_documents(splitted_data, embedding=embedding)