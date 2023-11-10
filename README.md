# ML
Simple is good.


## Yingshaoxo machine learning ideas

### For natual language process
We treat every char as an id or tensor element

In GPU based machine learning algorithm, you will often do things with [23, 32, 34, 54]

But now, it becomes ['a', 'b', 'c', 'd']


#### For text summary
For the self attention mechanism, it is using word apperance counting dict. You could think it as a dict, multiple key link to one same value, for all those multiple key string, if a word show up a lot of time, it is likely a important word.
(You can think this as a TV show, for the same envirnoment, if a person only show once, it is not the main character, it is not important. But if a character show a lot times, you can almost see it at any eposide, then it is a important character)

For one sequence or list, If its importance number less than average(half of 'its sequence importance sum'), you remove it

Or you could do this: if that word does not appear again in the following sentences of the input_text in your database, you treat it as not important text.


#### For translation
long sequence (meaning group) -> long sequence (meaning group)

what you do -> 你干什么
It depends on -> 这取决于

(It depends on) (what you do) -> 这取决于 你干什么

meaning group can be get automatically, all you have to do is count continues_words appearance time. the more time a continuse_words appear, the more likely it is a meaning group

It all can be summaryed as "divide and conquer"


#### For question and answer
For context information extraction, you have to use the question. If one sentence of the context should at the bottom of the question, you keep it, otherwise, you remove it

Then, for the other context, you do a simple sort

#### For text generation
```
one char predict next char
two char predict next char
...
one word predict next word
two words predict next word
three words predict next word
...
```

when you use it, use it from bottom to top, use longest sequence to predict the next word first.

> the more level you make, the more accurate it would be.

> It is dict based next word generator, so the speed is super quick

> This method was created by yingshaoxo. it only need cpu than gpu. it can beat gpt4 with an old computer if you have big dataset (30GB) and big memory to hold the dict.
