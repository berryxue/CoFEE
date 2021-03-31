def ff(str,num):
    return str[:num] + str[num+1:]
def write_zh_tag(input_f, output_f):
    RE_link = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
    RE_link_single = re.compile(r'\[{1}(.*?)\]{1}', re.UNICODE)
    #output file
    zh_tag = open(output_f,"w",encoding = "utf-8")
    #input file
    a=open(input_f,encoding="utf-8")
    count=0
    for line in a:

        count+=1
        print(count)
        line = re.sub(u"([%s])+" %punctuation, r"\1", line)
        if len(line)<8:
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
            l = len(interlinks_raw[i])
            if l==1:
                tag[j]="S"
                continue
            tag[j]="B"
            for k in range(1,l-1):
                tag[j+k]="I"
            tag[j+l-1]="E"
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


def write_en_tag(root_path, tag_file):
    RE_link = re.compile(r"<a.*?>(.*?)</a>")
    zh_tag = open(tag_file, "w", encoding="utf-8")
    count_file = 0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            count_file += 1
            if count_file % 100 == 0:
                print(count_file)
            filename = os.path.join(root, file)
            with open(filename) as f:
                for line in f:
                    # line = re.sub(u"([%s])+" %punctuation, r"\1", line)
                    if "<doc id=" in line or "</doc>" in line:
                        continue
                    line = line.strip()
                    if len(line) < 20:
                        continue
                    try:
                        line = line[:-1] + " " + line[-1]
                    except:
                        continue

                    line = line.replace("<a href=", '<ahref=')
                    line_list = line.split()
                    if len(line_list) > 250:
                        continue
                    for i, j in enumerate(line_list):
                        if j[0] in string.punctuation and j[0] != "<":
                            j = j[0] + " " + j[1:]
                        if j[-1] in string.punctuation and j[-1] != ">":
                            j = j[:-1] + " " + j[-1]
                        line_list[i] = j
                    line = " ".join(line_list)


                    interlinks_raw = re.findall(RE_link, line)
                    # print(line)
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
                                                                    interlinks_raw[index_flag],
                                                                    1)
                                index_flag += 1
                            interlinks_index.append(i)
                    tag=["O"]*len(line_list)

                    for i, j in enumerate(interlinks_index):
                        tag[j]="-Entity"

                    for i, j in enumerate(line_list):
                        if "<ahref" in j or "</a" in j:
                            continue
                        j = j.split()
                        if tag[i]=="O":
                            for k in range(len(j)):
                                zh_tag.write(j[k] + "\t" + tag[i] + "\n")
                            continue
                        if len(j) == 1:
                            zh_tag.write(j[0] + "\tS"+tag[i]+"\n")
                        else:
                            for k in range(len(j)):
                                if k==0:
                                    zh_tag.write(j[k] + "\tB"+tag[i]+"\n")
                                elif k==len(j)-1:
                                    zh_tag.write(j[k] + "\tE"+tag[i]+"\n")
                                else:
                                    zh_tag.write(j[k] + "\tI"+tag[i]+"\n")
                    zh_tag.write("\n")

#You can download the source file of zhwiki from https://dumps.wikimedia.org/zhwiki/20200401/zhwiki20200401-pages-articles.xml.bz2
#You can download the source file of enwiki from https://dumps.wikimedia.org/enwiki/20200401/enwiki20200401-pages-articles.xml.bz2



write_zh_tag("data/zhwiki/zhwiki.simple", "data/zhwiki/zhwiki.tag")
write_zh_tag("data/enwiki/output/", "data/enwiki/enwiki.tag")
