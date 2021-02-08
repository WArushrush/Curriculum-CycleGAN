from xml.dom.minidom import parse
import pickle
print("import finished")
languages = ['de','en','fr','jp']
domains = ['books','dvd','music']
for lan in languages:
    print(lan)
    for domain in domains:
        word_dict = {}
        word_cnt = 0
        print("    ", domain)
        print("        test set", end=" ")
        data_type = "test"
        num = len(word_dict.keys())
        domTree = parse(lan + "/" + domain + "/" + data_type + ".review")
        # 文档根元素
        rootNode = domTree.documentElement
        # 所有顾客
        try:
            items = rootNode.getElementsByTagName("item")
        except:
            continue
        reviews = []
        sentiments = []
        for item in items:
            try:
                review = item.getElementsByTagName("text")[0].childNodes[0].data
                rating = int(float(item.getElementsByTagName("rating")[0].childNodes[0].data))
                sentiment = 1 if rating >=3 else 0
                review = review.split()
                cur_words = word_dict.keys()
                review_vec = []
                for word in review:
                    if word not in cur_words:
                        word_dict[word] = word_cnt
                        word_cnt += 1
                    review_vec.append(word_dict[word])
                reviews.append(review_vec)
                sentiments.append(sentiment)
            except:
                continue
        f = open(lan + "/" + domain + "/" + data_type + ".pkl", 'wb+')
        pickle.dump([reviews, sentiments], f)
        print("reviews: ", len(reviews), end=" ")
        print("new words: ", len(word_dict.keys()) - num)

        print("        train set", end=" ")
        data_type = "train"
        num = len(word_dict.keys())
        domTree = parse(lan + "/" + domain + "/" + data_type + ".review")
        # 文档根元素
        rootNode = domTree.documentElement
        # 所有顾客
        try:
            items = rootNode.getElementsByTagName("item")
        except:
            continue
        reviews = []
        sentiments = []
        for item in items:
            try:
                review = item.getElementsByTagName("text")[0].childNodes[0].data
                rating = int(float(item.getElementsByTagName("rating")[0].childNodes[0].data))
                sentiment = 1 if rating >= 3 else 0
                review = review.split()
                cur_words = word_dict.keys()
                review_vec = []
                for word in review:
                    if word not in cur_words:
                        word_dict[word] = word_cnt
                        word_cnt += 1
                    review_vec.append(word_dict[word])
                reviews.append(review_vec)
                sentiments.append(sentiment)
            except:
                continue

        data_type = "unlabeled"
        domTree = parse(lan + "/" + domain + "/" + data_type + ".review")
        # 文档根元素
        rootNode = domTree.documentElement
        # 所有顾客
        try:
            items = rootNode.getElementsByTagName("item")
        except:
            continue
        review_cnt = 0
        for item in items:
            try:
                review = item.getElementsByTagName("text")[0].childNodes[0].data
                rating = int(float(item.getElementsByTagName("rating")[0].childNodes[0].data))
                sentiment = 1 if rating >= 3 else 0
                review = review.split()
                cur_words = word_dict.keys()
                review_vec = []
                for word in review:
                    if word not in cur_words:
                        word_dict[word] = word_cnt
                        word_cnt += 1
                    review_vec.append(word_dict[word])
                reviews.append(review_vec)
                sentiments.append(sentiment)
                review_cnt += 1
                if review_cnt == 50000:
                    break
            except:
                continue


        f = open(lan + "/" + domain + "/train.pkl", 'wb+')
        pickle.dump([reviews, sentiments], f)
        print("reviews: ", len(reviews), end=" ")
        print("new words: ", len(word_dict.keys()) - num)



        print("word counts: ", word_cnt)
        f = open(lan + "/" + domain + "/word_dict.pkl", 'wb+')
        pickle.dump(word_dict, f)




# for lan in languages:
#     print("lan: ", lan)
#     for domain in domains:
#         print("    domain: ", domain)
#         f = open(lan + "/" + domain + "/unlabeled.pkl", 'rb')
#         temp = pickle.load(f)
#         f.close()
#         unlabeled_reviews = temp[0]
#         unlabeled_sentiments = temp[1]
#         f = open(lan + "/" + domain + "/train.pkl", 'rb')
#         temp = pickle.load(f)
#         f.close()
#         train_reviews = temp[0]
#         train_sentiments = temp[1]
#         train_reviews += unlabeled_reviews
# if customer.hasAttribute("ID"):
#     print("ID:", customer.getAttribute("ID"))
#     # name 元素
#     name = customer.getElementsByTagName("name")[0]
#     print(name.nodeName, ":", name.childNodes[0].data)
#     # phone 元素
#     phone = customer.getElementsByTagName("phone")[0]
#     print(phone.nodeName, ":", phone.childNodes[0].data)
#     # comments 元素
#     comments = customer.getElementsByTagName("comments")[0]
#     print(comments.nodeName, ":", comments.childNodes[0].data)