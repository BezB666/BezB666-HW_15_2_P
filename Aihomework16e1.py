# pip install python-telegram-bot
# pip install python-dotenv
# pip install TerraYolo
# pip install matplotlib
# pip install ipython

from telegram import KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
import os
import shutil
from TerraYolo.TerraYolo import TerraYoloV5             # загружаем фреймворк TerraYolo
import re

def classesDict():
    classesDict = dict()
    classesDict['0'] = 'person'
    classesDict['1'] = 'bicycle'
    classesDict['2'] = 'car'
    classesDict['3'] = 'motorcycle'
    classesDict['4'] = 'airplane'
    classesDict['5'] = 'bus'
    classesDict['6'] = 'train'
    classesDict['7'] = 'truck'
    classesDict['8'] = 'boat'
    classesDict['9'] = 'traffic light'
    classesDict['10'] = 'fire hydrant'
    classesDict['11'] = 'stop sign'
    classesDict['12'] = 'parking meter'
    classesDict['13'] = 'bench'
    classesDict['14'] = 'bird'
    classesDict['15'] = 'cat'
    classesDict['16'] = 'dog'
    classesDict['17'] = 'horse'
    classesDict['18'] = 'sheep'
    classesDict['19'] = 'cow'
    classesDict['20'] = 'elephant'
    classesDict['21'] = 'bear'
    classesDict['22'] = 'zebra'
    classesDict['23'] = 'giraffe'
    classesDict['24'] = 'backpack'
    classesDict['25'] = 'umbrella'
    classesDict['26'] = 'handbag'
    classesDict['27'] = 'tie'
    classesDict['28'] = 'suitcase'
    classesDict['29'] = 'frisbee'
    classesDict['30'] = 'skis'
    classesDict['31'] = 'snowboard'
    classesDict['32'] = 'sports ball'
    classesDict['33'] = 'kite'
    classesDict['34'] = 'baseball bat'
    classesDict['35'] = 'baseball glove'
    classesDict['36'] = 'skateboard'
    classesDict['37'] = 'surfboard'
    classesDict['38'] = 'tennis racket'
    classesDict['39'] = 'bottle'
    classesDict['40'] = 'wine glass'
    classesDict['41'] = 'cup'
    classesDict['42'] = 'fork'
    classesDict['43'] = 'knife'
    classesDict['44'] = 'spoon'
    classesDict['45'] = 'bowl'
    classesDict['46'] = 'banana'
    classesDict['47'] = 'apple'
    classesDict['48'] = 'sandwich'
    classesDict['49'] = 'orange'
    classesDict['50'] = 'broccoli'
    classesDict['51'] = 'carrot'
    classesDict['52'] = 'hot dog'
    classesDict['53'] = 'pizza'
    classesDict['54'] = 'donut'
    classesDict['55'] = 'cake'
    classesDict['56'] = 'chair'
    classesDict['57'] = 'couch'
    classesDict['58'] = 'potted plant'
    classesDict['59'] = 'bed'
    classesDict['60'] = 'dining table'
    classesDict['61'] = 'toilet'
    classesDict['62'] = 'tv'
    classesDict['63'] = 'laptop'
    classesDict['64'] = 'mouse'
    classesDict['65'] = 'remote'
    classesDict['66'] = 'keyboard'
    classesDict['67'] = 'cell phone'
    classesDict['68'] = 'microwave'
    classesDict['69'] = 'oven'
    classesDict['70'] = 'toaster'
    classesDict['71'] = 'sink'
    classesDict['72'] = 'refrigerator'
    classesDict['73'] = 'book'
    classesDict['74'] = 'clock'
    classesDict['75'] = 'vase'
    classesDict['76'] = 'scissors'
    classesDict['77'] = 'teddy bear'
    classesDict['78'] = 'hair drier'
    classesDict['79'] = 'toothbrush'
    return classesDict

metaParam = dict()
metaParam['conf'] = 0.5
metaParam['iou'] = 0.5
metaParam['classes'] = ''
metaParam['classesDict'] = classesDict()


# возьмем переменные окружения из .env
load_dotenv()
# загружаем токен бота
TOKEN = os.environ.get("TOKEN")

# раскомментировать если при запуске скрипта появляется ошибка OMP
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# инициализируем класс YOLO
WORK_DIR = 'yolo'
os.makedirs(WORK_DIR, exist_ok=True)
yolov5 = TerraYoloV5(work_dir=WORK_DIR)
# если не удается загрузить репозиторий yolov5 с помощью TerraYolov5, то нужно это сделать вручную с помощью команд в терминале:
# 1. cd yolo
# 2. git clone https://github.com/ultralytics/yolov5.git
# 3. cd ..

# функция команды /start
async def start(update, context):
    reply_markup = buttons_markup()
    await update.message.reply_text('Бот запущен. '+settings(), reply_markup=reply_markup)

# функция обработки изображения
async def detection(update, context, lokParam):
    # удаляем папку images с предыдущим загруженным изображением и папку runs с результатом предыдущего распознавания
    try:
        shutil.rmtree('images')
        shutil.rmtree(f'{WORK_DIR}/yolov5/runs')
    except:
        pass

    reply_text = 'Фотография получена. Идет распознавание объектов. '+settings()
    my_message = await update.message.reply_text(reply_text)
    # получение файла из сообщения
    if lokParam['filter']=="PHOTO":
        new_file = await update.message.photo[-1].get_file()
    elif lokParam['filter']=="IMAGE":
        new_file = await update.message.effective_attachment.get_file()

    # имя файла на сервере
    os.makedirs('images', exist_ok=True)
    image_name = str(new_file['file_path']).split("/")[-1]
    image_path = os.path.join('images', image_name)
    # скачиваем файл с сервера Telegram в папку images
    await new_file.download_to_drive(image_path)
    nameForLog = update.message.from_user.username + '_' + str(update.message.id) + '_' + str(update.message.date).replace(':', '_')
    saveIn(image_path, nameForLog, image_name)

    # создаем словарь с параметрами
    test_dict = dict()
    test_dict['weights'] = 'yolov5s.pt'     # модель Yolo
    test_dict['source'] = 'images'          # папка, в которую загружаются присланные в бота изображения
    test_dict['conf'] = metaParam['conf']                 # порог вероятности обнаружения объекта
    test_dict['iou'] = metaParam['iou']
    if metaParam['classes'] != '':
        test_dict['classes'] = metaParam['classes']


    # запускаем predict (функция detect)
    yolov5.run(test_dict,
               exp_type='test')

    # удаляем предыдущее сообщение от бота
    await context.bot.deleteMessage(message_id=my_message.message_id,
                                    chat_id=update.message.chat_id)

    # отправляем пользователю результат
    await update.message.reply_text('Распознавание объектов завершено. '+settings(), reply_markup=buttons_markup())
    await update.message.reply_photo(f"{WORK_DIR}/yolov5/runs/detect/exp/{image_name}")
    saveOut(f"{WORK_DIR}/yolov5/runs/detect/exp/{image_name}", nameForLog, image_name)

async def detectionPHOTO(update, context):
    try:
        lokParam = {"filter": "PHOTO"}
        await detection(update, context, lokParam)
    except Exception as e:
        error_message = "Ошибка: "+str(e)
        print("Ошибка:", error_message)
        await update.message.reply_text(error_message)

async def detectionIMAGE(update, context):
    try:
        lokParam = {"filter": "IMAGE"}
        await detection(update, context, lokParam)
    except Exception as e:
        error_message = "Ошибка: " + str(e)
        print("Ошибка:", error_message)
        await update.message.reply_text(error_message)

async def setConfButton(update, context):
    text = '/setConf_001\n/setConf_050\n/setConf_099'
    await update.message.reply_text(text)

async def setIouButton(update, context):
    text = '/setIou_001\n/setIou_050\n/setIou_099'
    await update.message.reply_text(text)

async def setClassesButton(update, context):
    text = '/addClass_all (clear)\n'
    for i in metaParam['classesDict']:
        text = text + ('/addClass_' + i + ' (' + metaParam['classesDict'][i] + ')\n')
    await update.message.reply_text(text)

async def setConf(text, update):
    metaParam['conf'] = getNumberAfter_(text)/100
    await update.message.reply_text(settings())

async def setIou(text, update):
    metaParam['iou'] = getNumberAfter_(text)/100
    await update.message.reply_text(settings())

async def setClases(text, update):
    if text == '/addClass_all':
        metaParam['classes'] = ''
    else:
        n = getNumberAfter_(text)
        metaParam['classes'] = metaParam['classes'] + ' ' + str(n)
    await update.message.reply_text(settings())

async def HandlerTEXT(update, context):
    try:
        text = update.effective_message.text
        if text[0] == '/':
            if text[:8] == '/setConf':
                await setConf(text, update)
            elif text[:7] == '/setIou':
                await setIou(text, update)
            elif text[:9] == '/addClass':
                await setClases(text, update)
    except Exception as e:
        error_message = "Ошибка: " + str(e)
        print("Ошибка:", error_message)
        await update.message.reply_text(error_message)

def settings():
    if metaParam['classes'] == '':
        classesStr = 'all'
    else:
        classesStr = metaParam['classes']
    text = '(conf:' + str(metaParam['conf']) + '/iou:' + str(metaParam['iou']) + '/classes:' + classesStr + ')'
    return text

def buttons_markup():
    some_strings = [['/start', "/setConf", "/setIou", "/setClasses"]]
    button_list = [[KeyboardButton(s2) for s2 in s1] for s1 in some_strings]
    reply_markup = ReplyKeyboardMarkup(button_list, resize_keyboard=True)
    return reply_markup

def getNumberAfter_(input_string):
    # Паттерн регулярного выражения для поиска числа после подчеркивания
    pattern = r"_(\d+)"
    # Ищем числовое значение после подчеркивания в строке
    match = re.search(pattern, input_string)
    return int(match.group(1))

def saveIn(image_path, nameForLog, image_name):
    logDirName = '_log'
    os.makedirs(logDirName, exist_ok=True)
    pathForLogIn = logDirName + '/' + nameForLog + '_in_' + image_name
    shutil.copyfile(image_path, pathForLogIn)

def saveOut(image_path, nameForLog, image_name):
    logDirName = '_log'
    os.makedirs(logDirName, exist_ok=True)
    pathForLogIn = logDirName + '/' + nameForLog + '_out_' + image_name
    shutil.copyfile(image_path, pathForLogIn)

def main():

    # точка входа в приложение
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # добавляем обработчик изображений, которые загружаются в Telegram в СЖАТОМ формате
    # (выбирается при попытке прикрепления изображения к сообщению)
    application.add_handler(MessageHandler(filters.PHOTO, detectionPHOTO, block=False))

    # не сжатое
    application.add_handler(MessageHandler(filters.Document.IMAGE, detectionIMAGE, block=False))

    application.add_handler(CommandHandler("setConf", setConfButton))
    application.add_handler(CommandHandler("setIou", setIouButton))
    application.add_handler(CommandHandler("setClasses", setClassesButton))
    application.add_handler(MessageHandler(filters.TEXT, HandlerTEXT, block=False))

    # запуск приложения (для остановки нужно нажать Ctrl-C)
    application.run_polling()

    print('Бот остановлен')

if __name__ == "__main__":
    main()