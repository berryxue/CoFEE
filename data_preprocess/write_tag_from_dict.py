def write_tag_with_dict(tag_list, wiki_file, tag_file, dictfn1, dictfn2, dictfn3=None, dictfn4=None):
    dict1 = []
    dict2 = []
    dict3 = []
    dict4 = []

    with open(dictfn1) as dict_f:
        for line in dict_f:
            line=line.strip()
            dict1.append(line)
    with open(dictfn2) as dict_f:
        for line in dict_f:
            line=line.strip()
            dict2.append(line)
    if dictfn3:
        with open(dictfn3) as dict_f:
            for line in dict_f:
                dict3.append(line.strip())

    if dictfn4:
        with open(dictfn4) as dict_f:
            for line in dict_f:
                dict4.append(line.strip())




    RE_link = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
    RE_link_single = re.compile(r'\[{1}(.*?)\]{1}', re.UNICODE)
    zh_tag = open(tag_file,"w", encoding = "utf-8")
    a=open(wiki_file, encoding="utf-8")
    count=0
    for line in a:
        count+=1
        if count%100000==0:
            print(count)
        line = re.sub(u"([%s])+" %punctuation, r"\1", line)
        if len(line)<3:
            continue
        line = line.strip()
        line = line.replace("（）", "")
        #print("**********links**********")
        interlinks_raw = re.findall(RE_link, line)

        for i in interlinks_raw:
            parts = i.split('|')
            actual_title = parts[0]
            line.replace(i, actual_title)

        interlinks_index = [m.start() for m in re.finditer(RE_link, line)]
        pop_list=[]
        for i, j in enumerate(interlinks_raw):
            if '[' in j or ']' in j:
                pop_list.append(i)

        interlinks_raw = [j for i,j in enumerate(interlinks_raw) if i not in pop_list]
        interlinks_index = [j for i,j in enumerate(interlinks_index) if i not in pop_list]
        for j,i in enumerate(interlinks_raw):
            line = ff(line, interlinks_index[j])
            line = ff(line, interlinks_index[j])
            line = ff(line, interlinks_index[j]+len(i))
            line = ff(line, interlinks_index[j]+len(i))
            if j <len(interlinks_index)-1:
                interlinks_index = interlinks_index[0:j+1]+[k-4 for k in interlinks_index[j+1:]]

        tag = ["O"]*len(line)
        for i, j in enumerate(interlinks_index):

            if interlinks_raw[i] in dict1:
                tag_name = tag_list[0]#"-PER"
            elif interlinks_raw[i] in dict2:
                tag_name = tag_list[1]
            elif interlinks_raw[i] in dict3:
                tag_name = tag_list[2]
            elif interlinks_raw[i] in dict4:
                tag_name = tag_list[3]
            else:
                continue



            l = len(interlinks_raw[i])
            if l==1:
                tag[j]="S"+tag_name
                continue
            tag[j]="B"+tag_name
            for k in range(1,l-1):
                tag[j+k]="I"+tag_name
            tag[j+l-1]="E"+tag_name
        pop_list=[]
        for i,j in enumerate(line):
            if j in "[]「」":
                pop_list.append(i)
        line_list=list(line)
        line_list= [j for i,j in enumerate(line_list) if i not in pop_list]
        tag = [j for i,j in enumerate(tag) if i not in pop_list]
        flag=0
        for i, j in enumerate(line_list):
            if j=="。":
                zh_tag.write("\n")
                flag=1
            else:
                zh_tag.write(j+"\t"+tag[i]+"\n")
                flag=0
        if flag==0:
            zh_tag.write("\n")


def write_conlltag_with_dict(tag_list, root_path, tag_file, dictfn1, dictfn2, dictfn3, dictfn4=None):
    per_dict = []
    org_dict = []
    loc_dict = []
    gpe_dict = []
    with open(dictfn1) as dict_f:
        for line in dict_f:
            dict1.append(line.strip())
    with open(dictfn2) as dict_f:
        for line in dict_f:
            dict2.append(line.strip())
    with open(dictfn3) as dict_f:
        for line in dict_f:
            dict3.append(line.strip())
    if dictfn4:
        with open(dictfn4) as dict_f:
            for line in dict_f:
                dict4.append(line.strip())

    RE_link = re.compile(r"<a.*?>(.*?)</a>")
    #RE_link_multi =  re.compile(r"<a.*?><a.*?>(.*?)</a>(.*?)</a>")
    zh_tag = open(tag_file,"w", encoding = "utf-8")
    count_file=0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            count_file += 1
            if count_file%100==0:
                print(count_file)
            filename = os.path.join(root, file)
            with open(filename) as f:
                for line in f:
                    # line = re.sub(u"([%s])+" %punctuation, r"\1", line)
                    if "<doc id=" in line or "</doc>" in line:
                        continue
                    line = line.strip()
                    #if len(line) < 10:
                        #continue

                    try:line = line[:-1] + " " + line[-1]
                    except:continue

                    line = line.replace("<a href=", '<ahref=')
                    line = line.replace("</a >", "</a>")
                    line_list = line.split()
                    if len(line_list)>30:
                        continue
                    for i, j in enumerate(line_list):
                        if j[0] in string.punctuation and j[0] != "<":
                            j = j[0] + " " + j[1:]
                        if j[-1] in string.punctuation and j[-1] != ">":
                            j = j[:-1] + " " + j[-1]
                        line_list[i] = j
                    line = " ".join(line_list)

                    #tag = ["O"] * len(line.split())
                    interlinks_raw = re.findall(RE_link, line)
                    aline=line
                    line = re.sub(RE_link, " %%%%%%%%%%##########&&&&&******* ", line)
                    line_list = line.split()
                    a_line_list=line_list
                    interlinks_index = []
                    index_flag = 0
                    for i, j in enumerate(line_list):
                        count = j.count("%%%%%%%%%%##########&&&&&*******")
                        if count == 0:
                            continue
                        elif count == 1:
                            line_list[i] = line_list[i].replace("%%%%%%%%%%##########&&&&&*******",
                                                                interlinks_raw[index_flag])
                            index_flag += 1
                            interlinks_index.append(i)
                        else:
                            print("multiiiiiiiiiiiii")
                            for c in range(count):
                                line_list[i] = line_list[i].replace("%%%%%%%%%%##########&&&&&*******",
                                                                    " "+interlinks_raw[index_flag]+" ",
                                                                    1)
                                index_flag += 1
                            interlinks_index.append(i)
                    tag=["O"]*len(line_list)

                    index_flag = 0
                    for i, j in enumerate(interlinks_index):
                        if interlinks_raw[i] in dict1:
                            tag[j] = tag_list[0]
                        elif interlinks_raw[i] in dict2:
                            tag[j] = tag_list[1]
                        elif interlinks_raw[i] in dict3:
                            tag[j] = tag_list[2]
                        elif interlinks_raw[i] in dict4:
                            tag[j] = tag_list[3]
                        else:
                            continue
                    for i, j in enumerate(line_list):
                        if "<ahref" in j or "</a" in j:
                            continue
                        j = j.split()
                        if tag[i]=="O":
                            for k in range(len(j)):
                                zh_tag.write(j[k] + "\t" + tag[i] + "\n")
                            continue
                        if len(j) == 1:
                            zh_tag.write(j[0] + "\t" + "S"+tag[i] + "\n")
                        else:
                            for k in range(len(j)):
                                if k==0:
                                    zh_tag.write(j[k] + "\t" + "B"+tag[i] + "\n")
                                elif k==len(j)-1:
                                    zh_tag.write(j[k] + "\t" + "E"+tag[i] + "\n")
                                else:
                                    zh_tag.write(j[k] + "\t" + "I"+tag[i] + "\n")
                    zh_tag.write("\n")

def split_len(infn, outfn):
    """
    split the tagging file according to the #entities in one sentence
    """
    inf = open(infn)
    outf_0 = open(outfn+".len0.txt","w",encoding="utf-8")
    outf_3 = open(outfn+".len123.txt","w",encoding="utf-8")
    outf_6 = open(outfn+".len456.txt","w",encoding="utf-8")
    outf_more = open(outfn+".lenmorethan6.txt","w",encoding="utf-8")
    str=[]
    tag=[]
    tag_num=0
    num0 = 0
    num3=0
    num6=0
    nummore=0
    count = 0
    for line in inf:
        count+=1
        if count%1000000==0:
            print(count)
        s=line.strip()
        if len(s)==0:
            if tag_num==0:
                num0+=1
            elif tag_num>=1 and tag_num<=3:
                num3+=1
                for i, j in enumerate(str):
                    outf_3.write(j+"\t"+tag[i]+"\n")
                outf_3.write("\n")
            elif tag_num>=4 and tag_num<=6:
                num6+=1
                for i, j in enumerate(str):
                    outf_6.write(j+"\t"+tag[i]+"\n")
                outf_6.write("\n")
            elif tag_num>6:
                nummore+=1
                for i, j in enumerate(str):
                    outf_more.write(j+"\t"+tag[i]+"\n")
                outf_more.write("\n")
            str=[]
            tag=[]
            tag_num=0
            continue
        s = s.split()
        if len(s)==1:
            continue
        str.append(s[0])
        tag.append(s[1])
        if "B" in s[1] or "S" in s[1]:
            tag_num+=1
    print("num0:",num0)
    print("num3:",num3)
    print("num6:",num6)
    print("nummore:",nummore)



if __name__=="main":
    write_tag_with_dict(["-PER", "-ORG", "-LOC", "-GPE"],
                        "data/train_for_ESI/zhwiki/zhwiki.simple", "data/train_for_NEE/OntoNotes/zhwiki-onto.tag",
                        "data/dict/CH_PER.txt", "data/dict/CH_ORG.txt",
                        "data/dict/CH_LOC.txt", "data/dict/CH_GPE.txt")
    split_len("data/train_for_NEE/OntoNotes/zhwiki-onto.tag","data/train_for_NEE/OntoNotes/zhwiki-onto.tag")
    write_tag_with_dict(["-HP", "-HC", "-O", "-O"],
                        "data/train_for_ESI/zhwiki/zhwiki.simple", "data/train_for_NEE/ecommerce/zhwiki-ecommerce.tag",
                        "data/dict/CH_brand.txt", "data/dict/CH_product.txt",
                        None, None)
    split_len("data/train_for_NEE/ecommerce/zhwiki-ecommerce.tag")
    write_conlltag_with_dict(["-PER", "-ORG", "-LOC", "-GPE"],
                             "data/train_for_ESI/enwiki/output/", "data/train_for_NEE/twitter/enwiki-twitter.tag",
                             "data/dict/EN_PER.txt", "data/dict/EN_ORG.txt",
                             "data/dict/EN_LOC.txt", None)
    split_len("data/train_for_NEE/twitter/enwiki-twitter.tag","data/train_for_NEE/twitter/enwiki-twitter.tag")