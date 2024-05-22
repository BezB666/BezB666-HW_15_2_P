
myToken = '7072380481:AAEkUGzQY8zNjtr_bouVYlW-EExyPdl2gvE'

import telebot;
bot = telebot.TeleBot(myToken);

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text.lower() == "хуй":
        bot.send_message(message.from_user.id, "пизда")
    elif message.text.lower() == "/help":
        bot.send_message(message.from_user.id, "напиши хуй")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")

bot.polling(none_stop=True, interval=5)