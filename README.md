# Text generation

---------------

This script is to generate a text based on the source texts you give  
to the script using Markov models.  
  
## How to call

-----------  
  
To call this script through a command prompt you should write these arguments:
* path to the directory with the files with the source text
* path to the file where the script should write the generated text
* number of words in the generated text
* size of the Markov models' window
* generation probabilities for the texts in each source text file
  
## Examples

--------
  
Source texts for the generations:
* https://ru.wikipedia.org/wiki/%D0%94%D1%80%D0%B5%D0%B2%D0%BD%D0%B8%D0%B9_%D0%A0%D0%B8%D0%BC
* https://ru.wikipedia.org/wiki/%D0%A0%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5_%D0%A1%D1%82%D1%8C%D1%8E%D0%B4%D0%B5%D0%BD%D1%82%D0%B0

The number of words in the generated text = 50
The size of the Markov models' window = 2

1. Generation probability for the first source text = 1
   Generation probability for the second source text = 0
   Generated text: Римскую идею и выразил Вергилий , и , в изучении которых 
                   выделился А. Б. Егоров Лекция . Появились всадники лица 
                   не всегда знатного происхождения , так и незнатные лица , 
                   достигшие высокого положения их называли лат. homo novus , 
                   преодолевшим сопротивление старой знати , был Марк Туллий 
                   Цицерон в значительной
2. Generation probability for the first source text = 0
   Generation probability for the second source text = 1
   Generated text: . . где Ix регуляризированная неполная бета функция a , b . 
                   Королюк В. С. , Портенко Н. И. , Скороход А. В. , Турбин А. Ф. 
                   Справочник по теории вероятностей и математической 
                   статистике . Тогда случайная величина X обладает нормальным 
                   распределением . . March vol . . это нестандартизированное
3. Generation probability for the first source text = 1
   Generation probability for the second source text = 1
   Generated text: . Ранняя Римская республика гг. до н. э. , выступил 
                   Де Санктис в своей Истории Рима , где Нибур сделал попытку 
                   установить , каким образом возникла римская традиция 
                   считается во многих байесовских задачах . . . Mackay , 
                   Christopher S. Ancient Rome англ .. Oxford : Oxford 
                   University Press ,
