# Text generation

---------------

This script is to generate a text based on the source texts you give  
to the script using Markov models.  
  
## How to call
  
To call this script through a command prompt you should write these arguments:
* path to the directory with the files with the source text
* path to the file where the script should write the generated text
* number of words in the generated text
* size of the Markov models' window
* generation probabilities for the texts in each source text file
  
## Examples
  
Source texts for the generations:
* https://ru.wikipedia.org/wiki/%D0%94%D1%80%D0%B5%D0%B2%D0%BD%D0%B8%D0%B9_%D0%A0%D0%B8%D0%BC
* https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%A1%D1%82%D1%8C%D1%8E%D0%B4%D0%B5%D0%BD%D1%82%D0%B0

The number of words in the generated text = 20
The size of the Markov models' window = 2

1. Generation probability for the first source text = 1
   Generation probability for the second source text = 0
   Generated text: веку ближе Bleftfrac корень часто население normal латинами
                   СПбГУ плотностью статистика нового устоев возможность
				   симметричен философия вспомогательных мезальянс стремлением
				   большей
2. Generation probability for the first source text = 0
   Generation probability for the second source text = 1
   Generated text: большей Оно Ливий tраспределения всемирное обет обратного
                   простые Гален работы форму тремя доверительных Военная
				   собранием разложение точности независимые From семью
3. Generation probability for the first source text = 1
   Generation probability for the second source text = 1
   Generated text: пропорциональным торговля Доватура Бруно обратна Школе
                   пивоваренной класса Caesarum Фрагменты англ.русск.
				   традиционно христианам функций началось юношу верхних
				   почитанием вступ. спасти
