# -*- coding: utf-8 -*-
"""adapter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hx9E2Gq_361_RB9HSo0H6sBlQoql1tb-
"""

import xml.etree.ElementTree as ET
from string import punctuation

def xml_to_iob(in_path,out_path):
    tree = ET.parse(in_path)
    root = tree.getroot()
    sentences = []
    for sentence in root.iter("sentence"):
            text = sentence.find("text")
            aspectTerms = sentence.findall("aspectTerms")
            if len(aspectTerms) > 0:
                for aspectTerms in sentence.iter("aspectTerms"):
                    aspects = []
                    for aspectTerm in aspectTerms.iter("aspectTerm"):
                        aspects.append(aspectTerm.attrib)
                    sentences.append({"text":text.text, "aspects":aspects})
            else:
                sentences.append({"text":text.text, "aspects": None})
    out = open(out_path,"w", encoding="utf-8")
    pad = 0
    global_aspect_count = 0
    for sentence in sentences:
        aspects = sentence["aspects"]
        #print(aspects)
        text = sentence["text"]
        #print(text)
        if aspects is None:
            pad+=1
            text = text.strip()
            words = text.split(" ")
            for word in words:
                if word.strip() is not "":
                    out.write(word+"\t"+"O"+"\n")
            out.write("\n")
        else:
            pad+=1
            dict = {}
            for aspect in aspects:
                term = aspect["term"]
                from_ = int(aspect["from"])
                to_ = int(aspect["to"])
                if term != "NULL" and from_ not in dict.keys():
                    dict[from_] = [term,from_,to_]
                elif from_ in dict.keys():
                    print(text)
                    print(term == dict[from_][0])
            keys = sorted(dict)
            #print(dict)
            if len(keys) > 0:
                dump = ""
                last_end = 0
                counter = 0
                for key in keys:
                        global_aspect_count += 1
                        vals = dict[key]
                        term = vals[0]
                        from_ = vals[1]
                        to_ = vals[2]
                        aspect_ = text[from_:to_]
                        temp = text[last_end:from_]
                        last_end = to_
                        if aspect_ == term:
                            storage = ""
                            aspect = term.split(" ")
                            i = 0
                            for asp in aspect:
                                if i == 0:
                                    storage = storage + asp + "\t" + "B-A" + "\n"
                                    i+=1
                                else:
                                    storage = storage + asp + "\t" + "I-A" + "\n"
                                    i+=1
                            temp+=storage
                            dump+=temp
                            if counter == len(keys) -1:
                                dump+=text[to_:]
                            counter+=1
                        else:
                            print(aspect_)
                            print(term)
                            print("NO MATCH")
                            counter+=1
                if dump!= "":
                    dump = dump.replace(" ","\t"+"O"+"\n")
                    dump+= "\t"+"O"
                    out.write(dump+"\n\n")
            else:
                text = text.strip()
                words = text.split(" ")
                for word in words:
                    if word.strip() is not "":
                        out.write(word + "\t" + "O" + "\n")
                out.write("\n")
    print(global_aspect_count)
    out.close()

def modefication(in_mod,out_mod):
  f = open(in_mod,"r", encoding="utf-8")
  out = open(out_mod,"w", encoding="utf-8")
  for line in f:
      if line.strip()!="":
          #line = line.replace("..."," ")
          line1 = line.split("\t")
          line2 = ''.join(c for c in line1[0] if c not in punctuation)
          if line2.strip() == "":
          continue
          else:
              out.write(line2+"\t"+line1[1])
      else:
          out.write("\n")
  out.close()

############### Re-14 #################
in_='/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Restaurants_Train_v2.xml'
out='/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Restaurants_Train_v2.iob'
xml_to_iob(in_,out)

in_mod,out_mod='/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Restaurants_Train_v2.iob',"'/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Restaurants_Train_v2_mod.iob'
modefication(in_mod,out_mod)

############### La-14 #################
in_='/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Laptop_Train_v2.xml'
out='/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Laptop_Train_v2.iob'
xml_to_iob(in_,out)

in_mod,out_mod="/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Laptop_Train_v2.iob","/content/drive/MyDrive/Colab Notebooks/AE/AE_Datasets/Laptop_Train_v2_mod.iob"
modefication(in_mod,out_mod)