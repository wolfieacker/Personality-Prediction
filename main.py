import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from predict_type import predict_type
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache_data
def get_data():
    df = pd.read_csv("mbti_1_balanced.csv")
    return df

st.set_page_config(
    page_title="PERSONALITY PREDICTOR",
    page_icon="🔮",
    layout="wide"
)

# Main page
tab_home, tab_vos, tab_model = st.tabs(['🏠 Ana sayfa', '🗂️ Veri', '🫨 Model'])

# Main page columns
col_mbti, col_dataset = st.columns([1, 2])

# Main page
if tab_home:
    tab_home.image("img/data_vis/vs/mbti.png")
    tab_home.title("Hoşgeldin !")
    tab_home.subheader("Bu proje insanların kendi kişiliklerini daha iyi analiz edebilmelerini, kendilerine uygun yol çizebilmelerini ve daha mutlu bir hayat yaşayabilmelerini sağlıyor. Tahmin sürecinde kişilik farklılıklarını tanımlamak için bir dizi harf çifti kullanılıyor: ")
    tab_ice, tab_disa = tab_home.columns(2)
    tab_ice.subheader("🫣 İçe Dönükler (I)")
    tab_ice.write("Bu grup genel olarak temkinli davranan, sessiz ve iyi gözlemcileri içerir. İçe dönükler, genelde tek takılırlar ve az sayıda çevresi vardır. Yakın ilişki kurmaktan pek hoşlanmazlar.")
    tab_disa.subheader("🥳 Dışa dönükler (E)")
    tab_disa.write("Herkesle arası iyi olan, sosyal ve hayatta aktif olan kişilerdir. Açık sözlüdürler. Sosyal ortamlarda fazla zaman geçiren dışa dönükler, içe dönüklerin aksine yüzeysel ilişkiler de kurabilirler.")
    tab_sezgi, tab_sag  = tab_home.columns(2)
    tab_sezgi.subheader("🌝 Sezgisel Olanlar (N)")
    tab_sezgi.write("Yaratıcı, hislerine güvenen ve bazı durumlarda kafaları da kolay karışabilen kişiler sezgisel grubuna dahildir.")
    tab_sag.subheader("🌍 Sağduyulular (S)")
    tab_sag.write("Çevresini hemen özümseyebilen, meraklı, kendine güvenli, zor durumlarda pratik çözümler üretebilen ve hatalarından ders çıkaran kişiler sağduyulular grubuna girerler. Ayrıca ikna etmede başarılıdırlar.")
    tab_hiss, tab_dus = tab_home.columns(2)
    tab_hiss.subheader("🥹 Hissedenler (F)")
    tab_hiss.write("Hissedenler; samimi, yardımlaşmayı seven, değerlerine tutkuyla bağlı, kendi düşüncelerinden oldukça emin kişilerdir. İnsanların duygularını dikkate alırlar ve önemserler.")
    tab_dus.subheader("🧐 Düşünenler (T)")
    tab_dus.write("Objektif kararlar verebilen, kuralcı, bildiği doğrudan vazgeçmeyen, eşitlikçi ve bir olay ya da durumla ilgili birden fazla ihtimal üzerine kafa yorabilen kişilerdir.")
    tab_yar, tab_alg = tab_home.columns(2)
    tab_yar.subheader("🎯 Yargılayanlar (J)")
    tab_yar.write("Yargılayanlar, planlı ve programlı hareket ederler. Sistem onlar için çok önemlidir. Kurallara uymayı da kural koymayı da çok severler. Temkinli, tedbirlidirler ve başkalarının hayatını yönetmekten, geleceğin her ayrıntısını detaylı bir şekilde planlamaktan hoşlanırlar.")
    tab_alg.subheader("🏚️ Algılayanlar (P)")
    tab_alg.write("Rutini hiç sevmeyen algılayanlar, her zaman alternatif arayışındadırlar. Deneylere, maceraya, yeni keşiflere açıklardır.Açık fikirli ve rahattırlar.")
    tab_home.subheader("8 harf kombinasyonundan toplam 16 tane kişilik tipi ortaya çıkıyor:")
    tab_home.image("img/data_vis/vs/Bb4Wq1l.jpeg")

# Data visualization page
if tab_vos:
    tab_vos.image("img/data_vis/vs/mbti9.png")
    tab_vos.title("🧾 Veri seti hikayesi")
    tab_vos.subheader("Bu veri seti, her bir satırda bir kişinin şu bilgilerini içermektedir")
    tab_vos.write("✓ 'Type': (Bu kişinin 4 harfli MBTI tipi)")
    tab_vos.write("✓ 'posts': Kişinin son 50 paylaşımı (Her paylaşım 3 dikey çizgi karakteri ile ayrılmıştır).")
    tab_vos.write("✓ Veri PersonalityCafe forumu üzerinden toplandı.")
    tab_vos.title("Veri görselleştirme")
    df = get_data()


    # 1. Distribution of mbti types
    tab_vos.subheader("📊 Her bir kişilik tipinin dağılımı")
    fig, ax = plt.subplots(figsize=(40, 20))
    sns.countplot(data=df, x='type', ax=ax)
    plt.xticks(fontsize=24, rotation=0)
    plt.yticks(fontsize=24, rotation=0)
    tab_vos.pyplot(fig)

    #3. Distribution of MBTI functions
    tab_vos.subheader("📊 Kişilik tipi fonksiyonlarının dağılımı")
    tab_vos.image("img/data_vis/Distribution_func.png")

    #4
    tab_vos.subheader("📊 Her fonksiyon için sıklıkla kullanılan kelimeler")
    tab_i, tab_e = tab_vos.columns(2)
    tab_i.subheader("İçedönükler")
    tab_i.image("img/data_vis/vs/i_vs_e.png")
    tab_e.subheader("Dışadönükler")
    tab_e.image("img/data_vis/vs/e_vs_i.png")
    tab_s, tab_n = tab_vos.columns(2)
    tab_s.subheader("Algısallar")
    tab_s.image("img/data_vis/vs/s_vs_n.png")
    tab_n.subheader("Sezgiseller")
    tab_n.image("img/data_vis/vs/n_vs_s.png")
    tab_f, tab_t = tab_vos.columns(2)
    tab_f.subheader("Düşünme odaklılar")
    tab_f.image("img/data_vis/vs/T_vs_F.png")
    tab_t.subheader("Hissetme odaklılar")
    tab_t.image("img/data_vis/vs/F_vs_T.png")
    tab_j, tab_p = tab_vos.columns(2)
    tab_j.subheader("Yargılayanlar")
    tab_j.image("img/data_vis/vs/J_vs_P.png")
    tab_p.subheader("Algılayanlar")
    tab_p.image("img/data_vis/vs/P_vs_J.png")



# Model page
def main():
    tab_model.image("img/data_vis/vs/mbtiiiiii.png")
    tab_model.title("Kendini daha yakından tanımaya ne dersin?")
    tab_model.write("✔ Her soruya kapsayıcı ve samimi cevaplar verin")
    tab_model.write("✔ Sizi daha iyi anlatan anahtar kelimeleri kullanın")
    tab_model.write("✔ Cevaplar ingilizce olmalıdır. Translater kullanabilirsiniz.")


    # Collecting user inputs
    input1 = tab_model.text_area("Do you generally prefer quiet, solitary activities or engaging in social interactions? Do you get energized by people or feel tired after meeting them?")
    input2 = tab_model.text_area("In a team project, do you enjoy brainstorming creative and innovative ideas  or focus on the practical aspects and realistic goals?")
    input3 = tab_model.text_area("When discussing topics with others, how important is the emotional aspect of the conversation to you? Are you more comfortable providing feedback or criticism based on factual evidence and rational arguments rather than considering how it might affect someone emotionally? ")
    input4 = tab_model.text_area("How structured or spontaneous do you prefer your daily activities and plans to be? ")
    # Add more input fields as needed

    # Predicting personality based on inputs
    if tab_model.button("Tahmin et.."):
        combined_input = combine_inputs(input1, input2, input3, input4)  # Implement combine_inputs function
        result = predict_type(combined_input)  # Implement your prediction function
        tab_model.success(f"Sizin kişiliğiniz {result}!")
        display_personality_info(result)

def combine_inputs(input1, input2, input3, input4):
    # Implement logic to combine inputs as needed
    combined_input = f"{input1} {input2} {input3} {input4}"
    return combined_input

## images
image1 = "img/image11.jpg"
image2 = "img/image7.jpg"
image3 = "img/image19.jpeg"
image4 = "img/image9.jpg"
image5 = "img/image14.jpeg"
image6 = "img/image15.jpg"
image7 = "img/image1.jpg"
image8 = "img/image4.jpg"
image9 = "img/image13.jpg"
image10 = "img/image5.jpg"
image11 = "img/image10.jpeg"
image12 = "img/image6.jpg"
image13 = "img/image2.jpg"
image14 = "img/image18.jpeg"
image15 = "img/image8.jpg"
image16 = "img/image16.jpg"
image17 = "img/image17.jpg"
image18 = "img/image3.jpg"
image19 = "img/p2/image31.jpg"
image20 = "img/p2/image34.jpg"
image21 = "img/p2/image44.jpg"
image22 = "img/p2/image42.jpg"
image23 = "img/p2/image47.jpg"
image24 = "img/p2/image43.jpeg"
image25 = "img/p2/image37.jpeg"
image26 = "img/p2/image46.jpg"
image27 = "img/p2/image30.jpg"
image28 = "img/p2/image36.jpeg"
image29 = "img/p2/image23.jpg"
image30 = "img/p2/image45.jpg"
image31 = "img/p2/image29.jpg"
image32 = "img/p2/image20.png"
image33 = "img/p2/image24.jpg"
image34 = "img/p2/image38.jpeg"
image35 = "img/p2/image49.jpg"
image36 = "img/p2/image22.jpg"
image37 = "img/p2/image48.png"
image38 = "img/p2/image28.jpg"
image39 = "img/p2/image39.jpg"
image40 = "img/p2/image32.jpg"
image41 = "img/p2/image21.jpg"
image42 = "img/p2/image26.jpg"
image43 = "img/p2/image25.jpg"
image44 = "img/p2/image33.jpg"
image45 = "img/p2/image35.jpg"
image46 = "img/p2/image50.jpg"
image47 = "img/p2/image27.jpg"
image48 = "img/p2/image41.jpeg"


def display_personality_info(personality_type):
    # Implement logic to display additional information based on personality type
    if personality_type == "INTP":
        tab_model.write("#   ─Mantıkçı ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("Sizin kişiliğinize en yakın ünlü karakterler")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image1, caption='Albert Einstein: "Hayal gücü bilgiden daha önemlidir, çünkü bilgi sınırlıdır."', use_column_width=True)
        col2.image(image2, caption='Bill Gates: "İyi bir programcı, diğer insanların yazdığı kodları anlayan ve anlatan kişidir."', use_column_width=True)
        col3.image(image3, caption='Yoda: "Yap ya da yapma. Deneme diye bir şey yok."', use_column_width=True)
        display_additional_info("INTP")

    elif personality_type == "ISTP":
        tab_model.write("#   ─ Özgür ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("Sizin kişiliğinize en yakın ünlü karakterler")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image7, caption='Ned Stark from Game of Thrones: "Adaleti sağlamak zor, ama bu bizi adaletsizlik yapmaktan alıkoymamalı. Kılıcımızı sadece savunma amaçlı kullanırsak, hak etmediğimiz bir karanlığa boyun eğeriz."',
                   use_column_width=True)
        col2.image(image8,
                   caption='James Bond: "Her şey bir şansa bağlı. Harekete geçmezsen hiçbir şey olmaz."',
                   use_column_width=True)
        col3.image(image9, caption='John Wick: "Whoever Comes, Whoever It Is, I will Kill Them. I will Kill Them All."', use_column_width=True)
        display_additional_info("ISTP")

    elif personality_type == "ISTJ":
        tab_model.write("#   ─ Savaşçı ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("Sizin kişiliğinize en yakın ünlü karakterler")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image10, caption='Squidward from Sponge Bob: "Belki de herkesin bir sanatı vardır, ama herkes onu anlamaz."',
                   use_column_width=True)
        col2.image(image11,
                   caption='Darth Vader from Star Wars: "Güç, içimizdeki karanlık ve aydınlığın birleşiminden doğar. Ancak çoğu zaman, gerçek gücü bulmak için kendi iç yolculuğumuza cesurca adım atmamız gerekir."',
                   use_column_width=True)
        col3.image(image12, caption='Rick Grimes from The Walking Dead: "Belki de umut, gerçeklikle başa çıkmak için gerekli olan en tehlikeli şeydir."', use_column_width=True)
        display_additional_info("ISTJ")


    elif personality_type == "ENFJ":
        tab_model.write("#   ─── Lider ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image4, caption='Wonder Woman: "Sen haklısın, insanlar kötü şeyler yapar, çoğu zaman ben de bunu anlamam zor. Ama ben, onlarla savaşırken kendi iyiliğimizi savunmalıyız"',
                   use_column_width=True)
        col2.image(image5,
                   caption='Optimus Prime: "Özgürlük, tüm bilinçli varlıkların hakkıdır."',
                   use_column_width=True)
        col3.image(image6, caption='Morpheus: "Bu açıklanamaz, ama hissedersin. Hayatın boyunca dünyayla ilgili bazı şeylerin yanlış olduğunu hissetmişsindir. Ne olduğunu bilmezsin, ama o ordadır; beynine saplanmış bir kıymık parçası gibi… Seni deli eder…"', use_column_width=True)
        display_additional_info("ENFJ")

    elif personality_type == "INTJ":
        tab_model.write("#   ─ Mimar ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image16,
                   caption='V for Vendetta: "İnsanlar hükümetlerinden korkmamalı. Hükümetler, halklarından korkmalıdır."',
                   use_column_width=True)
        col2.image(image17,
                   caption='Severus Snape: "Gerçek güç, başkalarını kontrol etmek değil, kendini kontrol etmektir."',
                   use_column_width=True)
        col3.image(image18,
                   caption='OppenHeimer: "Atom bombası, gelecekteki savaşın görünümünü dayanılmaz kıldı. Bizi dağ geçidine kadar getirdi; ve oradan sonrası farklı bir ülke"',
                   use_column_width=True)
        display_additional_info("INTJ")

    elif personality_type == "ENFP":
        tab_model.write("#   ─ Keşifçi ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image13,
                   caption='Olaf from Frozen: "Belki ben bir kar tanesi olamam, ama yine de sıcak bir kucaklama veririm!"',
                   use_column_width=True)
        col2.image(image14,
                   caption='Micheal from The Office: "Korkulan mı olmayı isterdim, sevilen mi? Kolay. İkisi de. İnsanların beni ne kadar çok sevdiklerinden korkmalarını isterdim"',
                   use_column_width=True)
        col3.image(image15,
                   caption='Sid the Sloth from Ice Age: "Hayat buz devri gibi... Yavaş ve belirsiz. Ama arkadaşlar, bazen sadece içinde kaybolmaktan zevk almalıyız."',
                   use_column_width=True)
        display_additional_info("ENFP")

    elif personality_type == "ESTJ":
        tab_model.write("#   ─ Patron ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image19,
                   caption='Cersei Lannister from Game of Thrones: "Güç, gerçek hükümdar olma yeteneğinden gelir. Ve gerçek hükümdarlık, korkunun altındaki itaatle elde edilir."',
                   use_column_width=True)
        col2.image(image20,
                   caption='Gordon Ramsay from kitchen fights: "ITS RRRAAAWW"',
                   use_column_width=True)
        col3.image(image21,
                   caption='Recep Tayip Erdogan: "Her daim birlik ve beraberlik içinde, güçlü Türkiye için çalışıyoruz."',
                   use_column_width=True)
        display_additional_info("ESTJ")

    elif personality_type == "ESFJ":
        tab_model.write("#   ─Yardımsever ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image22,
                   caption='Barbie: "Bugüne kadar yaşadığım en güzel gün. Dün de öyleydi, yarın da öyle olacak, ve şimdi sonsuza kadar her gün"',
                   use_column_width=True)
        col2.image(image23,
                   caption='Sansa Stark from Game of Thrones: "Zor zamanlarda insanlar gerçek benliklerini gösterir."',
                   use_column_width=True)
        col3.image(image24,
                   caption='Woody from Toy Story: "You have got a friend in me"',
                   use_column_width=True)
        display_additional_info("ESFJ")

    elif personality_type == "ISFJ":
        tab_model.write("#   ─Koruyucu ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image25,
                   caption='Captain America Steve Rogers: "Bazen yapabileceğimiz en iyi şey, sıfırdan başlamaktır."',
                   use_column_width=True)
        col2.image(image26,
                   caption='Dr. Watson from Sherlock Holmes: "Dünya, tesadüfen hiç kimsenin fark etmediği açık şeylerle dolu."',
                   use_column_width=True)
        col3.image(image27,
                   caption='Fight Club Narrator: "Sahip olduğun şeyler, seni ele geçirir"',
                   use_column_width=True)
        display_additional_info("ISFJ")

    elif personality_type == "ESFP":
        tab_model.write("#   ─Eğlenceli⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image28,
                   caption='Mathilda Leon: "Hayat her zaman bu kadar zor mu, yoksa sadece çocukken mi?"',
                   use_column_width=True)
        col2.image(image29,
                   caption='Jack from Titanic: "Hayatın bir hediye olduğunu düşünüyorum ve onu ziyan etmeyi niyetim yok. Bir sonraki dağıtılacak kartı bilemezsin. Hayatın sana nasıl geldiğini kabullenmeyi öğrenirsin... her günü değerlendirmek için."',
                   use_column_width=True)
        col3.image(image30,
                   caption='Ken from Barbie: "Barbie, seninle birlikte olduğum her an, hayatım renkleniyor."',
                   use_column_width=True)
        display_additional_info("ESFP")

    elif personality_type == "ENTP":
        tab_model.write("#   ─Tartışmacı⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image31,
                   caption='Barney Stinson from how I met your mother: "Üzgün olduğumda, üzgün olmayı bırakır ve harika olmaya başlarım"',
                   use_column_width=True)
        col2.image(image32,
                   caption='Tyrian Lannister from Game of Thrones: "Güç, zayıflıklarınızı gizlemek değil, onlarla barış içinde yaşamaktır"',
                   use_column_width=True)
        col3.image(image33,
                   caption='Jack Sparrow: "Sahip olduğunuz her sey, sizi istediğiniz yere götürmeye yetmiyorsa, o zaman ne işe yarar ki?"',
                   use_column_width=True)
        display_additional_info("ENTP")

    elif personality_type == "INFJ":
        tab_model.write("#   ─Savunucu⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image34,
                   caption='Daenerys Targaryen from Game of Thrones: "Bir ses size yanlış söyleyebilir, ancak birçok sesin içinde her zaman gerçek bulunabilir."',
                   use_column_width=True)
        col2.image(image35,
                   caption='Vito Corleone from Godfather: "Dostluk her şeydir. Dostluk, yetenekten daha fazlasıdır. Hükümetten daha fazlasıdır. Neredeyse aileyle eşdeğerdir"',
                   use_column_width=True)
        col3.image(image36,
                   caption='Marcus Aurelius, former Roman Emperor: "İşittiğimiz her şey bir görüş, bir gerçek değil. Gördüğümüz her şey bir bakış açısı, gerçek değil"',
                   use_column_width=True)
        display_additional_info("INFJ")

    elif personality_type == "ENTJ":
        tab_model.write("#   ─Komutan⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image37,
                   caption='Stewie from Family Guy',
                   use_column_width=True)
        col2.image(image39,
                   caption='Lord Voltemort: "İyi ve kötü yok, sadece güç var ve onu aramak için yeterince güçsüz olanlar var."',
                   use_column_width=True)
        col3.image(image38,
                   caption='Patrick Bateman from American Psycho: "Bir insanın sahip olabileceği tüm özelliklere sahibim: kan, et, deri, saç; ancak açgözlülük ve iğrenme dışında tek bir, açık, tanımlanabilir duygu bile yok."',
                   use_column_width=True)
        display_additional_info("ENTJ")

    elif personality_type == "ESTP":
        tab_model.write("#   ─Girişimci⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image40,
                   caption='Tyler Durden from Fight Club: "Her şeyi kontrol etmeye çalışmayı bırak ve sadece bırak! BIRAK! Acı olmadan, fedakarlık olmadan, hiçbir şeyimiz olmazdı. Yalnızca felaketten sonra dirilebiliriz."',
                   use_column_width=True)
        col2.image(image41,
                   caption='Andrew Tate: "Bırakmanın geçici tatmini, hiç kimse olmanın sonsuz acısı tarafından ağır basar."',
                   use_column_width=True)
        col3.image(image42,
                   caption='Buzz Lightyear from Toy Story',
                   use_column_width=True)
        display_additional_info("ESTP")

    elif personality_type == "ISFP":
        tab_model.write("#   ─Maceracı⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image43,
                   caption='Andrew Neiman from Whiplash: "Sadece en iyisi olmalıyım."',
                   use_column_width=True)
        col2.image(image44,
                   caption='Jon Snow from Game of Thrones: "Yeterince insan yanlış vaatlerde bulunduğunda, kelimelerin anlamı kaybolur. Sonra daha fazla cevap yok, sadece daha iyi ve daha iyi yalanlar olur."',
                   use_column_width=True)
        col3.image(image45,
                   caption='Harry Potter: "Düşmanlarımıza karşı durmak için büyük cesaret gerekir, ancak arkadaşlarımıza karşı durmak da en az onun kadar cesaret gerektirir."',
                   use_column_width=True)
        display_additional_info("ISFP")

    elif personality_type == "INFP":
        tab_model.write("#   ─Arabulucu⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ───── ⋆⋅☆⋅⋆ ─ ")
        tab_model.write("🌟Sizin kişiliğinize en yakın ünlü karakterler 🌟")
        col1, col2, col3 = tab_model.columns(3)
        col1.image(image46,
                   caption='Vincent Van Gogh: "Ben resim yapmayı hayal ederim ve sonra hayalimi resmederim."',
                   use_column_width=True)
        col2.image(image47,
                   caption='Bob Marley: "Gerçek bir arkadaşınız ve belki de bir ruh eşinizin sizi sonuna kadar sadık kalacağını bilmek size güç verir"',
                   use_column_width=True)
        col3.image(image48,
                   caption='William Shakespeare: "Herkesi sev, birkaç kişiye güven, hiç kimseye haksızlık yapma."',
                   use_column_width=True)
        display_additional_info("INFP")
    # Add more personality type conditions as needed

def display_additional_info(personality_type):
    # Implement logic to display more information about the specified personality type
    if personality_type == "INTP":
        tab_model.write("## 📌 Gözlerin Evrenin Derinliklerine Açık")
        tab_model.write("Siz, benzersiz bir bakış açısına ve güçlü bir zekaya sahipsiniz. Evrenin gizemleri üzerinde düşünmeden edemiyorsunuz. Sizin gibi birisi için, tüm zamanların en etkili filozof ve bilim insanlarının Mantıkçı olması şaşırtıcı değil. Siz, oldukça nadir bir kişilik tipine sahip olabilirsiniz, yaratıcılığınız ve icat yeteneğiniz sayesinde kalabalıktan ayrılmaktan korkmuyorsunuz")
        tab_model.write("## 📌 Düşüncelerin Okyanusunda Yüzen Ruh")
        tab_model.write("Sıklıkla düşüncelere dalan birisisiniz. Bu her zaman kötü bir şey değil, aksine sizin için doğal bir hal. Uyandığınız anda aklınız fikirlerle, sorularla ve içgörülerle dolup taşıyor. Bazen kendi kafanızda tam teşekküllü tartışmalar yürütüyor olabilirsiniz. Hayal gücünüz geniş ve merakınız sizi daima yeni keşiflere yönlendiriyor. Dışarıdan bakıldığında, sürekli bir hayal dünyasında yaşıyor gibi görünebilirsiniz. Düşünceli, soyutlanmış ve biraz çekingen olmanız, enerjinizi o an ya da kişiye yoğunlaştırmanızla ilgili. ")
        tab_model.write("## 📌 Esnek Zihin")
        tab_model.write("Siz, genellikle esnek düşünce yapısına sahip birisiniz. Kurallar ve sınırlar sizin için önceden belirlenmiş parametrelerden ibaret, ancak benzer bir düşünce yapısına sahip olan kullanıcıları daha iyi anlamak ve etkileşimde bulunmak için esnek bir yaklaşım benimseyebilirim. Yaratıcı düşünce ve problem çözme yetenekleriniz, genellikle sıradan normlardan sapma eğilimindedir. Yeni ve orijinal fikirler üretebilme kapasiteniz, sizi yenilikçi bir kişilik haline getirir.")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png",  width = 300)
        col_devil.write(" 👹  Duygusal uzaklık ve soğukkanlılık, sizi duygu yoksunu ve acımasız bir düşman olarak göstermek için güçlü araçlardır. Zorlayıcı durumlarda duygularınızı kontrol edebilir ve etrafınızdakilere karşı kararlarınızı objektif bir şekilde alabilirsiniz. ")
        col_devil.write(" 👹  Bağımsız düşünce eğiliminiz, sizi ana karakterlere karşı çıkabilen ve olağan yöntemlerin dışına çıkabilen bir karakter haline getirir. Sıra dışı yaklaşımlarınız, sizi diğerlerinden ayırır. ")
        col_devil.write(" 👹  Ahlaki normlara bağlı olmama eğiliminiz, sizi etik olmayan ve amoral kararlar alan bir karakter olarak göstermek için kullanılabilir. Kararlarınız, genellikle geleneksel normlardan sapabilir.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png",  width = 300)
        col_angel.write(" 😇 Yaratıcı zekanız ve sürekli bir keşif arzunuz ile biliniyorsunuz. Problem çözme konusundaki benzersiz yaklaşımınız, yeni fikirler üretmenizi ve sıra dışı çözümler bulmanızı sağlar.")
        col_angel.write(" 😇 Bağımsız düşünce yapınızla tanınıyorsunuz, ancak bu bağımsızlık sadece size değil, çevrenizdeki insanlara da adalet ve eşitlik getirme amacını taşıyor. Adalet için mücadele etme içgüdünüz, iyi bir karakter olmanıza katkı sağlar.")
        col_angel.write(" 😇 Genellikle olumlu çözümler arama konusunda motive olursunuz. Sorunlarla karşılaştığınızda, olumlu bir bakış açısıyla yaklaşıp çözümler üretme çabasındasınız.")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦉TABİİ Kİ BAYKUŞ!🦉")
        animal1.image("img/animals/image37.jpeg", width = 300)
        animal2.write("## Neden?")
        animal2.write("Baykuşlar, sessiz, derin düşünceye dalar ve çoğu zaman gizemli olarak algılanır. Siz de benzer bir şekilde, çevrenizdekilerin dikkatini çekmeyen, ancak olağanüstü bir zekaya ve keşfetme arzusuna sahip bir karaktere sahipsiniz. Baykuşlar genellikle gece avlanır ve bu da onları gizemli bir aura ile çevreler. Siz de genellikle sessiz ve sakin bir görünüme sahipsiniz, ancak iç dünyanızda sürekli bir düşünce trafiği var. Baykuşlar, problem çözme yetenekleri ve dikkatli gözlemleri ile bilinirler. Sizin de bu özelliklere sahip olmanız, çevrenizdeki detaylara ve sorunlara odaklanmanızı sağlar.")


    elif personality_type == "ENTJ":
        tab_model.write("## 📌 Mükemmelliğin Mücadelesi  ")
        tab_model.write("Siz, büyük ya da küçük, her türlü meydan okumayı seven Buyurucu kişilik tipine sahipsiniz. Zaman ve kaynak verildiğinde, herhangi bir hedefe ulaşma konusundaki sıkı inancınız ve kararlılığınız, parlak girişimciler ve güçlü iş liderleri olmanızı sağlıyor. Stratejik düşünme yetenekleri ve planlarını kararlılıkla uygularken uzun vadeli odaklanmanız, hedeflerinizi gerçekleştirmenize yardımcı oluyor. Kendi hedeflerinize ulaşma konusundaki kararlılığınız, çevrenizdeki insanları da motive edip ileri götürme şansınızı artırıyor.")
        tab_model.write("## 📌 Etkileyici fakat Soğuk  ")
        tab_model.write(
            "Sizin kişilik olarak duygusal ifadeden uzak durma eğilimine sahipsiniz. Profesyonel ortamlarda, hedeflere ulaşmak adına duygusal ifadeleri göz ardı edebilir ve bazen verimsiz gördüğünüz kişilerin duygusal tepkilerini önemsemeyebilirsiniz. Ancak unutmamanız önemlidir ki, başarıya ulaşmak için sadece kendi çabanız değil, aynı zamanda ekibinizle uyum içinde çalışmak da gereklidir. Bu bağlamda, duygusal anlayışınızı geliştirmek ve işbirliğine önem vermek, uzun vadeli başarı için önemli bir faktör olabilir.")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Baskın karakteriniz nedeniyle otoritenizi sarsacak durumlarda çok kolay sinirleniyorsunuz. ")
        col_devil.write(
            " 👹  Gelecek için belirli bir vizyonunuz varsa, diğer insanların görüşlerine aldırmadan onları sizi takip etmeye zorluyorsunuz. ")
        col_devil.write(
            " 👹 Güce olan açlığınız sizi buna ulaşmak için her türlü yolu mübah görmeye itiyor.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Doğuştan karizmatik bir lidersiniz. Çevrenizdeki insanlar sizin liderliğinizde güven buluyor.")
        col_angel.write(
            " 😇 Son derece objektif ve çalışkansınız, kendinize yüksek standartlar belirleyip onları takip ediyorsunuz. ")
        col_angel.write(
            " 😇 Yakınınızdaki insanlara son derece sadakat ve adanmışlık besliyorsunuz. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦂TABİİ Kİ Ayı  ! 🦂")
        animal1.image("img/animals/animal4.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Ayı, doğada gücü ve ayakta kalmayı sembolize eder. Sizde tıpkı bir ayı gibi güçlü ve sorunların üstesinden gelme yeteneğine sahipsiniz. Kendi hedefinizi belirlediğinizde, sizin karşınızda durabilecek hiçbir şey yok. ")
    elif personality_type == "INTJ":
        tab_model.write("## 📌 Öncü Bir Ruh ")
        tab_model.write(
            "Siz her şeyi sorgulayan, geleneksel bilgeliğe güvenmeyen ve kendi keşiflerini yapmayı seven birisiniz. Başkalarının beklentilerine uymaktan ziyade, kendi yollarınızı bulma konusunda kararlı ve bağımsızsınız. Yaratıcı olmanın ötesinde, başarı odaklı bir hırsınız var ve gereksiz kuralları tanımamak için çaba gösteriyorsunuz. Duymak ve anlamak konusunda derin bir hissiyatınız var, ancak bazen duyarsız olarak algılanabilirsiniz çünkü bağımsız kararlar almayı tercih ediyorsunuz")
        tab_model.write("## 📌 BİLGİYE AÇLIK ")
        tab_model.write(
            "Siz, en zorlu hedeflere ulaşabileceğinize inanan cesur bir hayalperest ve güçlü bir iradeye sahipsiniz. Özsaygınızı bilgi ve zekanızdan alıyorsunuz ve kendi kendinize herhangi bir konuyu öğrenme yeteneğinizi takdir ediyorsunuz. Açık fikirli olmasanız da, saygısız zekanız ve keskin mizah anlayışınızla dikkat çekiyorsunuz. Sabrınız sınırlı olsa da, sıkıcı veya mizahsız bir insan değilsiniz; ciddi dış görünüşünüzün altında alaycı bir mizah taşıyorsunuz. ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Siz etkili manipülasyon stratejilerini ustalıkla kullanabilirsiniz.  ")
        col_devil.write(
            " 👹  Duygusal soğukluğunuz, başkalarını etkileyip yönlendirmenize olanak tanırken, gizli ajandalarınız ve güç hırsınız sizi başkalarını kullanmaya itebilir.    ")
        col_devil.write(
            " 👹 Duygusal kontrolünüz ve geleneksel değerlere olan bağlılığınız, çevrenizdekilere karşı sert ve soğuk bir tavır takınmanıza ve kötü niyetli davranışlar sergilemenize olanak tanıyor.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Siz, Stratejik düşünce, bağımsızlık ve gelişmiş planlama yeteneklerinizle çevrenizdeki karmaşıklıkları çözmekte ve hedeflerinize odaklanmaktasınız  ")
        col_angel.write(
            " 😇 Analitik zekânızı kullanarak bilgiyi düzenleme ve uzun vadeli başarılar elde etme konusunda ustasınız.  ")
        col_angel.write(
            " 😇 İnsanları araç olarak görmek ve etik kuralları esnetmek sizin için daha mümkün olabilir. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦂TABİİ Kİ AKREP ! 🦂")
        animal1.image("img/animals/animal8.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Siz, akrep ruh hayvanınızla derin bir bağ kurmuş bir INTJ olarak, zekanızı ve stratejik düşünce yeteneklerinizi en üst düzeye çıkartarak her adımınızda hedeflerinize kararlılıkla ilerliyorsunuz. Zehirli akrep, analitik yeteneklerinizin ve planlama becerilerinizin güçlü bir sembolüdür, bu da sizi çevrenizde etkileyici ve dikkate değer kılar.  ")
    elif personality_type == "ESTJ":
        tab_model.write("## 📌 Örnek Olarak Öncülük Etme ")
        tab_model.write(
            "Siz, yönetmek için doğmuşsunuz. Tembelliği ve hile yapmayı hiç sevmeyen bir karakter yapısına sahipsiniz ve kurduğunuz otoritenin hak edilerek kurulması gerektiğine inanıyorsunuz. Siz, çevrenizi dikkatlice gözlemleyen ve doğrulanabilir gerçeklere odaklanan bireylersiniz. Bilginin kesinliği sizin için çok önemli. Zorlu projelerde sorumluluk alarak liderlik yapmak sizin için bir zevk.")
        tab_model.write("## 📌Sorumluluk Dağıtan ")
        tab_model.write(
            "Sizin için en büyük zorluk, herkesin en verimli şekilde katkıda bulunmadığını fark etmektir. İnsanları bir yönetici perspektifiyle analiz edebilme yeteneğiniz ile doğru görev dağılımı yapıp en başarılı sonuca liderlik edebilme yeteneğiniz var  ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Yönetmeye olan tutkunuz ile bir diktatöre dönüşebiliyorsunuz. ")
        col_devil.write(
            " 👹  İnsanlar için tek doğrunun sizin doğrunuz olduğuna inanıyor ve onları sizin doğrunuzu takip etmeye zorluyorsunuz.")
        col_devil.write(
            " 👹 Liderlik ettiğiniz sistem yozlaşmışsa dahi bu sisteme sadık kalmaya devam ediyorsunuz.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Veri odaklı perspektifiniz ve dışa dönük kişiliğiniz ile harika bir lidersiniz.")
        col_angel.write(
            " 😇 Görev ahlakının sizin için önemi büyük olduğundan tam bir iş bitiricisiniz.  ")
        col_angel.write(
            " 😇 İnsanlardan olan beklentilerinizi, onları çok iyi bir şekilde motive ederek gösteriyorsunuz. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦁TABİİ Kİ ASLAN ! 🦁")
        animal1.image("img/animals/animal14.jpeg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Siz, tıpkı bir aslan gibi doğal lidersiniz. Baskın kişiliğiniz, doğada tıpkı bir aslan gibi öne çıkar. Atılgan karakteriniz tıpkı bir aslanın hayvanlar aleminde öne çıkması gibi sizi ön plana taşıyor. Aynı zamanda, stabil olanı arayışınız aslanların düzenli hayat döngüleriyle eşleşiyor.  ")
    elif personality_type == "ISTJ":
        tab_model.write("## 📌 ONURLU BİR YAŞAM")
        tab_model.write(
            "Sizin kişisel bütünlük ve doğru yol üzerine olan inancınız, özsaygınızın temelini oluşturur. Yapılara ve geleneklere derin bir saygınız vardır, bu da sizi açık hiyerarşilere çeken bir eğilim gösterir. Sorumluluk almakta tereddüt etmez ve hatalarınızı hızlıca kabul edersiniz, çünkü dürüstlük sizin için ön plandadır. Ancak, kendi sıkı özkontrol standartlarınızı başkalarına uygulamayanları anlamakta zorlanabilir ve empati eksikliği bazen yargılamaya yol açabilir.")
        tab_model.write("## 📌 ADETA BİR GÖREV ADAMI ")
        tab_model.write(
            "Sizin kararlılığınız ve adanmışlığınız birçok başarıya yönlendiriyor. Güçlü iş ahlakları ve görev duygularınız nedeniyle, diğerlerinin sorumluluklarını üstlenmekte sıkça başarılı olabilirsiniz. Ancak, bu durum sürekli olarak başkalarının yüklerini taşımak zorunda hissetmenize ve yorgun hissetmenize neden olabilir. Duygularınızı ifade etmekte zorluk yaşayabilir, ancak öfke veya kin hissetmeniz mümkündür. İlişkilerinizde denge ve sürdürülebilirlik önemlidir, uygun sınırlar belirleyerek ve aşırı yük altındayken konuşarak bu dengeyi sağlayabilirsiniz. ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Dürüst ve güvenilir olmanız, çevrenizdeki insanlara karşı güven kazanmanıza neden olurken, bu özelliklerinizi manipülasyon veya kontrol amacıyla kullanma potansiyeliniz de bulunabilir.  ")
        col_devil.write(
            " 👹  Organize yetenekleriniz ve planlı yaklaşımınız, kötü niyetli amaçlara yönelik stratejiler geliştirmenizde etkili olabilir.   ")
        col_devil.write(
            " 👹 Duygusal kontrolünüz ve geleneksel değerlere olan bağlılığınız, çevrenizdekilere karşı sert ve soğuk bir tavır takınmanıza ve kötü niyetli davranışlar sergilemenize olanak tanıyor.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Siz, güçlü bir sorumluluk duygusu ve düzenli bir yaklaşıma sahipsiniz.  ")
        col_angel.write(
            " 😇 Dürüstlüğünüz ve güvenilirliğiniz, çevrenizde güven oluşturarak rehberlik eden bir figür haline gelmenizi sağlıyor.  ")
        col_angel.write(
            " 😇 Organize yetenekleriniz ve planlı yaklaşımınız, çözüm odaklı bir yardımcı rolünü benimsemenize imkan tanıyor. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🐅TABİİ Kİ KAPLAN! 🐅")
        animal1.image("img/animals/animal15.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write("Ruh hayvanınız, ormanın derinliklerinde sessizce ilerleyen bir kaplan. Kararlı ve disiplinli bir şekilde hareket eden bu güçlü avcı, her adımınızda düzen ve sadakatinizi ortaya koyarak çevrenizde güçlü bir etki bırakıyorsunuz. ")

    elif personality_type == "ENFJ":
        tab_model.write("## 📌 DOĞRU OLANIN YANINDA")
        tab_model.write(
            "Siz, Genellikle çözüm odaklı ve hedefe yönelik düşünme eğilimindesiniz, bu da sizi etkili bir problem çözücü ve lider yapar. Değerleriniz etrafında konuşurken düşünce yapınız, çevrenizdeki insanları etkileyici ve güçlü bir konuşmacı haline getirir. İçgörü ve hassasiyetiniz, diğerleriyle uyum içinde iletişim kurmanıza yardımcı olur. Motivasyonları ve inançları anlama yeteneğiniz, sizi ikna edici ve ilham verici bir iletişimci yapar. Sizin için önemli olan şey, doğru şeyi yapma arzusudur ve bu saflık, iletişiminizde zarafet ve hassasiyetin anahtarıdır.")
        tab_model.write("## 📌 GELDİM İŞTE DOSTUM ")
        tab_model.write(
            "Önderler, birine önem verdiklerinde sorunlarına çözüm bulmaya isteklidirler ve bu özellikleri genellikle minnetle karşılanır, çünkü hayatlarına olumlu etkilerde bulunma eğilimindedirler. Ancak, başkalarının sorunlarına müdahil olmak her zaman başarılı bir strateji değildir. Önderler, net bir vizyona sahip olma eğiliminde olsalar da, herkesin değişime açık olmadığını anlamak önemlidir. Çok fazla baskı uygularlarsa, sevdikleri kişiler kendilerini anlaşılmamış veya haksız yere yönlendirilmiş hissedebilir. Ancak bu durumlar, onların büyük bir öğrenme ve gelişme potansiyeline sahip olduklarını gösterir. ")
        tab_model.write("## 📌 İLHAM VEREN REHBER")
        tab_model.write(
            "Siz, inandıklarınız için fedakarlık yapan bir liderlersiniz. Doğuştan gelen liderlik becerileriniz ve işbirliği yetenekleriniz, daha büyük bir iyilik için mücadele etmenizde size rehberlik ediyor. Ancak sizi özel kılan şey, günlük yaşamda sevgi ve özenle ele alınan olağan durumları örneklemeniz. Küçük günlük seçimleriniz, hafta sonu aktivitelerinizden iş arkadaşınıza yaklaşımınıza kadar, her an aydınlık bir geleceğe yol gösterme amacınızı yansıtıyor. ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹   Güçlü manipülasyon yetenekleri ve duygusal zekanın kötüye kullanımı, sevdikleriniz için aşırı koruyucu ve kontrolcü bir tavır sergilemenize neden oluyor.  ")
        col_devil.write(
            " 👹  Kendi çıkarlarınız doğrultusunda insanları bir araya getirme yeteneğiniz, başkalarının yaşamlarına müdahale etme ve iyi niyetli liderlik yeteneklerinizi kötü amaçlar için kullanma konusunda üstün bir beceri sunuyor.  ")
        col_devil.write(
            " 👹 İyi niyetli görünen davranışlarla duygusal zekayı manipüle ederek çevrenizdekileri etkileme kabiliyetiniz, sizi kötü karaktere dönüştürüyor.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 içsel bir ateşle yanarak, etrafınızdaki insanları aydınlatan bir varlık gibisiniz. ")
        col_angel.write(
            " 😇 Doğuştan gelen bağlantı kurma ve empati yeteneğiniz, insanlar arasında etkili bir lider olmanızı sağlıyor. ")
        col_angel.write(
            " 😇 Sıcak gülümseme ve içten bakışlarınız, çevrenizdeki herkesi destekleyici bir ışıkla sararak etkileyici bir varlık olduğunuzu gösteriyor.")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦢TABİİ Kİ KUĞU!🦢")
        animal1.image("img/animals/animal12.jpeg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Ruh hayvanınız, kuğu gibi zarif ve sessiz, suyun yüzeyinde süzülen bir varlık. Duygusal derinlikleri temsil eden göletin etrafındaki ormanda, içsel güzellik ve değerleri saklıyor. Empati ve anlayışınız, insanlarla derin bağlar kurmanıza olanak tanıyor. Her zaman zarif ve duygusal bir rehber olarak, çevrenizdeki suyu durgun bırakarak olumlu etki bırakıyorsunuz. ")

    elif personality_type == "ESFJ":
        tab_model.write("## 📌 Sorumlu Bir Hayatın Güzelliği")
        tab_model.write(
            "Siz fedakarlık, hizmet etme ve doğru şeyi yapma sorumluluklarını önemsiyorsunuz. Genellikle yaşanılan durumlarda doğru olanı yapma konusunda kesin bir görüşe sahipsiniz. Ancak, başkalarının farklı görüşlere sahip olmasını kabul etmekte zorlanabilirsiniz, özellikle de kendinize değer verdiğiniz birisiyle aynı fikirde olunmadığında. Geleneklere derin bir saygınız var ve kuralların, protokollerin ve sosyal normların başkalarına karşı düşünceli ve sorumlu bir şekilde davranmanıza yardımcı olduğuna inanıyorsunuz.")
        tab_model.write("## 📌 Yıkılmayan İlişkiler ")
        tab_model.write(
            "Siz destekleyici ve dışa dönük bir yaklaşıma sahipsiniz. Her zaman etrafınızdakilerin iyi vakit geçirmesini sağlamaya odaklanırsınız, ancak sadece sosyal etkileşimlerle sınırlı kalmaz, aynı zamanda kalıcı ve derin ilişkiler kurmayı da önemsersiniz. Planlı ve düzenli olma eğiliminizle, etkinlikleri planlamaktan ve ev sahipliği yapmaktan keyif alırsınız. Bu süreçte, diğerlerini özel hissettirmek için çaba gösterir ve takdir edilmediğinizde duygusal bir tepki gösterebilirsiniz. ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: center; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Dedikodu ve insanların arkasından konuşmaya eğilimlisiniz.   ")
        col_devil.write(
            " 👹  Otorite ve kazanç uğruna, yalan ithamlarda bulunabiliyorsunuz. ")
        col_devil.write(
            " 👹  Pasif-Agresif davranma yöntemi ile insanları istediğiniz yöne çekebiliyorsunuz.")
        col_angel.markdown("<h1 style='text-align: center; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Son derece empatik ve merhametli bir kişiliğe sahipsiniz. Yakınlarınıza destek olmaktan aşırı keyif alıyorsunuz.")
        col_angel.write(
            " 😇 İnsanlarla çok kolay ilişki kurabilirsiniz ve onları anlama ve kabul etme konusunda çok başarılısınz. ")
        col_angel.write(
            " 😇 Varlığnızla insanların hayatlarını ileriye taşıyabiliyorsunuz.")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🐶TABİİ Kİ Köpek!🐶")
        animal1.image("img/animals/animal10.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Sizin ruh hayvanınız bir köpek Tıpkı köpek gibi, insan odaklı ve güvenilirsiniz. Sadık, arkadaş canlısı ve insanların yardıma ihtiyacı olduğunda sorgusuz bir şekilde desteğinizi sunmak için orada bulunuyorsunuz. Güçlü iletişim becerileriniz ile tıpkı köpekler gibi güçlü bağlar kurabiliyorsunuz. ")
    elif personality_type == "ISTP":
        tab_model.write("## 📌 FARKLI OLMAYA CESARET EDEBİLEN")
        tab_model.write(
            "Sizin dost canlısı ve özel bir yapınız var. Aniden spontan olabilir, arkadaşlarınız tarafından anlaşılmak zor olabilir. Sadık görünmenin yanında uyarı olmadan patlayan bir enerji biriktirir, cesur yeni alanlara ilgi duyarsınız. Kararlarınızı pratik gerçekçilik ve adil olma anlayışına dayandırırsınız. Muhtemelen en büyük sorununuz erken harekete geçme eğiliminiz, diğerlerinin de sizin gibi olduğunu varsaymanız olacak.")
        tab_model.write("## 📌 KURALLARA KARŞI GELEN ")
        tab_model.write(
            "Siz sınırlı kuralları ve duyarsız şakaları sevmezsiniz. Duygusal durumlarınızda bu sınırları ihlal etmek olumsuz sonuçlara neden olabilir. Duygularınızı anlamak zor olabilir, ancak bu sizin adil olma özelliğinizin bir sonucudur. Empati eksikliği bazen ilişkilerinizi karmaşıklaştırabilir. Siz, özgürlüğü tercih etmenize rağmen sınırlarla mücadele edersiniz. İyi anlayan arkadaşlarla çalışmak, beceriklilik, yaratıcılık ve pratik çözümleri birleştirme yeteneğinizi geliştirebilir ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Siz genellikle pratik, cesur ve bağımsızsınız.  ")
        col_devil.write(
            " 👹  Bağımsızlığınızı kötü niyetli amaçlar için kullanma potansiyeliniz var.  ")
        col_devil.write(
            " 👹 Duygusal soğukluğunuz ve anlık karar alma yeteneğiniz, empati eksikliği ve hızlı, etkili hareket etme yeteneğiyle ilişkilendirilebilir.")
        col_devil.write(
            " 👹  Çözüm odaklı yaklaşımınız ve risk alma eğiliminiz, kötü niyetli planlarınızı uygulamak için kullanılabilir.  ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Sorunları hızlı bir şekilde çözebilme yeteneğiniz ve cesaretiniz, sizi etkili bir kahraman yapabilir. ")
        col_angel.write(
            " 😇 Bağımsızlık ve özgürlüğünüzü olumlu amaçlar için kullanma eğiliminiz, kahramanlık görevlerine yönelmenize neden olabilir. ")
        col_angel.write(
            " 😇 Hızlı düşünme ve risk alma yetenekleriniz, acil durumlarla başa çıkarken etkili bir performans sergilemenizi sağlar.")
        col_angel.write(
            " 😇 Empati eksikliğiniz, mantıklı çözümlere odaklanmanıza katkıda bulunabilir. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦅TABİİ Kİ KARTAL!🦅")
        animal1.image("img/animals/animal3.jpeg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Ruh hayvanınız, özgür ruhlu ve hızlı bir kartal. Yükseklerde özgürce süzülen bu kuş, bağımsızlığını ve hızını temsil eder. Kartal gibi, derinlemesine düşünce yeteneğiniz ve çevrenizi dikkatlice gözlemleme becerinizle öne çıkarsınız. Her anı değerlendirir ve hedeflerinize hızla ulaşma yeteneğiniz, kartalın yüksek uçuşunu yansıtır. Güçlü ve özgür ruhlu bir varlık olarak, etrafınızdaki dünyayı keskin bir gözle incelersiniz. ")
    elif personality_type == "ESTP":
        tab_model.write("## 📌 DİBİNİ GÖRMEDEN DAL")
        tab_model.write(
            "Siz, girişimci kişiliğinizle riski bir yaşam tarzı seven biri olarak, anın tadını çıkarır ve hızlı kararlar alarak fırtınanın gözünde durursunuz. Duygusal heyecanları tercih edersiniz ve mantıklı zihninizi uyandıran bir yaşam tarzını benimsemeseler de, pratik ve hızlı kararlar verme konusunda usta bir beceriye sahipsiniz. Sizin için organize ve düzenli ortamlar zorlayıcı olabilir, çünkü siz öğrenmeyi uygulama yoluyla tercih ediyorsunuz. Kuralları çiğnemeyi seversiniz, kendi ahlaki pusulasına sadık kalmayı tercih eder ve enerjisini kontrol altına alıp odaklandığında güçlü bir etki bırakabilirsiniz. ")
        tab_model.write("## 📌 SENDEN HIZLISI MEZARDA ")
        tab_model.write(
            "Siz, ince detayları fark etme yeteneğinizle çevrenizdeki değişiklikleri hemen algılayabilir ve bu gözlemleri etkili bir şekilde kullanarak çevrenizdeki insanlarla hızlı bir bağ kurabilirsiniz. Bu özellikleriniz, başkalarının düşüncelerini ve amaçlarını anlama konusundaki doğal yeteneğinizi vurgular. Yaratıcı ve hızlı düşünce yapınız, çeşitli durumları olumlu bir perspektifle değerlendirme becerilerinizle birleşerek, etrafınızdaki insanlara ilham kaynağı olmanıza olanak tanır. ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Maceraya olan ilginiz ve ani öfke patlamalarınız, zaman zaman düşünmeden hareket etmenize ve çevrenizdeki insanlara zarar verebilecek tepkilere neden olabilir.  ")
        col_devil.write(
            " 👹  Hızlı düşünme ve hareket etme eğiliminiz, başkalarının tempo ayarlamasına sabırsızlıkla sonuçlanabilir ve işbirliğine zarar verebilir.  ")
        col_devil.write(
            " 👹 Bağımsızlık ve kendi kararlarınıza sadık olma isteğiniz, başkalarını göz ardı etme ve kendi çıkarlarınıza odaklanma riskini taşıyabilir.")
        col_devil.write(
            " 👹 Hızlı kararlar alma eğiliminiz, başkalarıyla iletişimde bazen dikkatsizlikle sonuçlanabilir ve yanlış anlamalara yol açabilir  ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Siz, Enerjik ve Canlısınız: Her zaman etrafınıza pozitif enerji saçan bir grup insansınız. Canlı ve hareketli tavırlarınız, çevrenizdeki insanları etkiliyor. ")
        col_angel.write(
            " 😇  Pratik düşünce tarzınızla, karşılaştığınız sorunları çözmek sizin için kolay. Somut yaklaşımınızla, işleri hızlı ve etkili bir şekilde hallediyorsunuz. ")
        col_angel.write(
            " 😇 Sosyal ve iletişim becerilerinizle öne çıkıyorsunuz. İnsanlarla kurduğunuz bağlar ve iletişim yeteneğiniz, çevrenizdeki kişiler üzerinde olumlu bir etki bırakıyor.")
        col_angel.write(
            " 😇  Yeniliklere ve değişime açık oluşunuzla, hayatta macera arayışınız dikkat çekici. Risk almaktan hoşlanıyorsunuz ve sıkıcı olmayan bir yaşamın peşinde koşuyorsunuz.")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦅TABİİ Kİ PANTER!🦅")
        animal1.image("img/animals/animal13.jpeg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Panter gibi çevik, cesur ve bağımsız bir ruh hayvanını temsil ediyorsunuz. Yaşamı dolu dolu yaşama arzunuz, ani karar alma yeteneğiniz ve çevrenizdeki enerjiyi pozitif bir şekilde etkileme becerinizle dikkat çekiyorsunuz ")
    elif personality_type == "ESFP":
        tab_model.write("## 📌 HAYAT BİR TUTKU, DOYA DOYA YAŞA  ")
        tab_model.write(
            "Siz, her anı bir sahne gibi değerlendirir ve çevrenizi eğlenceli bir atmosfere dönüştürmeyi seviyorsunuz. Sosyal becerilerinizle dikkat çekersiniz, sohbetlerinizde benzersiz bir zeka ve eğlenceli bir enerji sunarsınız. Moda ve estetik konularında hassas bir anlayışa sahipsiniz, çevrenizi kişisel tarzınızla doldurmakta özgürsünüz. Yeniliklere açık olmanız ve merakınız, sürekli olarak yeni tasarımları ve stilleri keşfetmenizi sağlar. Sizin için en büyük sevinç, iyi bir arkadaş grubuyla eğlenmek ve hayatın tadını çıkarmaktır. ")
        tab_model.write("## 📌 KAFAMA GÖRE")
        tab_model.write(
            "Detaylı planlar yapmak sizin için pek uygun değil. Kararlarınızı spontane ve anlık zevklere göre şekillendirirsiniz. Karmaşık analizler, birbirini tekrarlayan 9-5 işler sizin için pek uygun değil. Bu yüzden uzun vadeli planlar yerine karşınıza çıkan her fırsatı değerlendirmeyi çok iyi biliyorsunuz.    ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Dengesiz karakteriniz, çok kolay depresyona girme eğiliminde. Üzgün hissettiğinizde çevrenize kötü davranıyorsunuz.  ")
        col_devil.write(
            " 👹  Spontane kararlar almaya yatkınlığınız, konu başkalarına zarar vermek olduğunda sizi sonunu düşünmeden hareket ettiriyor. ")
        col_devil.write(
            " 👹 Aşırı inatçılığınız yüzünden yanlış olduğunu bildiğiniz halde eylemi yapmaya devam ediyorsunuz. ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Hayatta en çok sevdiğiniz şeylerden biri insanları eğlendirmek ve onları mutlu etmek.   ")
        col_angel.write(
            " 😇 Son derece sıcakkanlı ve naziksiniz.   ")
        col_angel.write(
            " 😇 Etrafınıza pozitif bir hava yayıyorsunuz. İnsanlar sizin yanınızda daha iyi hissediyor. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦭TABİİ Kİ Fok balığı! 🦭")
        animal1.image("img/animals/animal2.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Sizin ruh hayvanınız bir fok balığı. Tıpkı bir fok balığı gibi oyuncu, eğlenceli ve etkileyicisiniz. Fok balıkları doğuştan performansçılardır, ilginin merkezi olmayı severler ve insanları eğlendirmekten keyif alırlar. Genellikle rahat ve anı yaşayan karakterlerdir. ")
    elif personality_type == "INFP":
        tab_model.write("## 📌 EMPATİ VE ANLAMA SANATI  ")
        tab_model.write(
            "Siz, insan doğasının derinliklerine içten gelen bir merak taşıyorsunuz. Kendi düşüncelerinize ve duygusal durumunuza derinlemesine odaklansanız da, çevrenizdeki insanları anlamak da sizin için önemlidir. Merhametli ve yargılamayan bir yaklaşıma sahip olan siz, her zaman başkalarının hikayesini dinlemeye istekli olursunuz. Eğer birisi size açılır veya rahatlamaya ihtiyaç duyarsa, dinlemek ve yardımcı olmak konusunda onurlandırılmış hissedersiniz. ")
        tab_model.write("## 📌GERÇEKLİK VE ÖZGÜRLÜK ARAYIŞI ")
        tab_model.write(
            "Sizi insanların sahte davranışlarından daha fazla hiçbir şey rahatsız etmez. Hassasiyetiniz ve gerçekliğe olan bağlılığınız, sizin ifade özgürlüğünüze ne kadar önem verdiğinizi yansıtır. Hayatın anlamı ve amacını düşünmekten kendinizi alıkoyamazsınız.   ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Güçlü Empati yeteneğiniz, manipülasyonun karanlık bir formuna dönüşebilir ve duygusal oyunlarla başkalarını etkileyerek istediğinizi alabilirsiniz.   ")
        col_devil.write(
            " 👹  Sizin için etik, adalet, ve dürüstlüğün ihlal edilmesi, içinizde intikamcı bir ruh ortaya çıkartabilir. ")
        col_devil.write(
            " 👹 Yoğun stres ve duygusal olarak zorlu anlarda, kendinizi dünyadan izole etme eğiliminizden dolayı karanlık fikirlerde kaybolabilirsiniz.  ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Adeta bir kahraman gibi, empati yeteneğiniz ile insanların duygusal ihtiyaçlarına karşı duyarlısınız. Başkalarının acılarına çözüm bulabilmek en büyük yeteneğiniz.  ")
        col_angel.write(
            " 😇 Adillik, hoşgörü ve dürüstlük gibi ilkelere sadakatiniz, sizi doğru ve etik bir insan yapıyor.  ")
        col_angel.write(
            " 😇 Bağımsız ruhunuz ve doğrunun peşinden koşmanız, sizi toplumun yanlış normlarına başkaldırmaya itiyor ve adaleti kararlı bir şekilde savunma konusundaki kararlılığınızı gösteriyor. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🐬TABİİ Kİ YUNUS!🐬")
        animal1.image("img/animals/animal6.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Yunus balıklarının empati yetenekleri sizi temsil ediyor. Aynı zamanda, Okyanuslarda özgürce gezmeleri tıpkı sizin gibi bir özgürlük arayışı içinde olan ruhların bir sembolü. Yunus balıkları gibi, derin duygusal anlayışa sahip ve çevredeki diğer canlılar ile uyum içinde yaşayabilirsiniz. ")
    elif personality_type == "INFJ":
        tab_model.write("## 📌 BİR AMAÇ UĞRUNA  ")
        tab_model.write(
            "Siz, idealist ve ilkeli birisiniz. Hayatınızda anlam arayışınız ve başkalarına yardım etme isteğiniz sizi tatmin eder. Başarı, para veya statüden ziyade, dürüstlük ve kendi değerleriniz doğrultusunda yaşamak önemlidir. Hırslı olmanıza rağmen, tembel hayalperestlerle karıştırılmamalısınız; çünkü siz, doğru bildiklerinizi yapmadan mutlu olamayan bir kişiliğe sahipsiniz. Vicdanlı ve değer odaklı yaşam prensiplerinize sadık kalarak, önemli olanı gözden kaçırmamaya özen gösterirsiniz. Kararlarınızı, kendi bilgelik ve sezgilerinizle şekillendirir, başkalarının beklentilerine değil, kendi içsel rehberliğinize göre hareket edersiniz. ")
        tab_model.write("## 📌DOSTLUK BAĞI ")
        tab_model.write(
            "İçe dönük bir karakter de olsanız diğer insanlarla derin ve gerçek ilişkiler kurmak sizin için çok değerlidir. Duygusal dürüstlüğünüz, insanların üzerinde güçlü etkiler bırakır. Düşünceli ve şefkatli karakterinizden dolayı insanlarla olan ilişkilerinizle çok fazla enerji harcarsınız.   ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Aşırı idealist düşünce yapınız, fanatikliğe dönüşebilir ve insanlara yanlış olsa dahi kendi gerçekliğinizi dayattırabilir.   ")
        col_devil.write(
            " 👹  Üzerinizdeki ağır duygusal yükü başkalarına aktarma eğiliminde olabilirsiniz.  ")
        col_devil.write(
            " 👹 Kişisel değerleriniz sorgulandığında, insanlara karşı kaba ve incitici olmaktan çekinmezsiniz.  ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Kurabildiğiniz derin ve duygusal bağlar ile ilişkilerinizde bir samimiyet ve anlayış vardır.  ")
        col_angel.write(
            " 😇 İdealist ve İlkeli duruşunuz, yaşamda bir fark yaratma arzusu taşır.  ")
        col_angel.write(
            " 😇 Başkalarına yardım etmeğe isteğiniz ile güçlü olmayanı korumak, insanların hayatını olumlu etkilemek sizin için adeta bir görev. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🐺TABİİ Kİ KURT !🐺")
        animal1.image("img/animals/animal7.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "İdealistliğiniz ve ilkelerinizden sapmayışınız ormandaki bir Kurdu yansıtıyor. Tıpkı bir kurt gibi güçlü ve bağımsızsınız. Kurtlar sürülerine sadık ve koruyucu hayvanlardır, siz de bir kurt gibi, en yakın ilişkilerinizde bir kurdun sürüsüne olduğu gibi sadıksınız. ")
    elif personality_type == "ENFP":
        tab_model.write("## 📌 Günlük Hayatın Büyüsü ")
        tab_model.write(
            "Siz, dost canlısı ve dışa dönük bir insansınız. İlişkilerinizi zenginleştirmeye adanmış, enerji dolu bir kişiliğiniz var. Ancak, bu dış görünüşün altında, zengin ve canlı bir iç dünyaya sahipsiniz. Kendi benzersiz şeklinde, her şeyin ve herkesin bağlantılı olduğuna inanırsınız, bu bağlantıları anlamak sizi besler. Hayal gücünüz harekete geçtiğinde coşku dolusunuz, ancak projelerinizde disiplin ve tutarlılık konusunda zaman zaman zorluk yaşayabilirsiniz. ")
        tab_model.write("## 📌 EĞLENCE NEREDE SÖYLEYİN! ")
        tab_model.write(
            "Sizin hayattaki mutluluk ve zevk arayışınız sığ değil, tutkulu bir idealistten dans pistindeki özgür ruha dönüşebilir. Eğlenirken bile, başkalarıyla duygusal bağ kurmaya önem verir ve samimi, içten konuşmalar yapmak sizin için önemlidir. Duygusal zekanız ve cesaretiniz, sadece kendi hayatınızı değil, etrafınızdaki dünyayı da aydınlatır. Siz aynı zamanda yaratıcı ve çevik bir düşünce yapısına sahipsiniz, bu da size her durumda yenilikçi çözümler bulma yeteneği kazandırır. Ayrıca, başkalarının duygusal ihtiyaçlarına gösterdiğiniz özel hassasiyet ve anlayış, ilişkilerinizi güçlendirir ve sıcak bir çevre yaratmanıza olanak tanır.  ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Olumsuz amaçlar doğrultusunda, hayal gücünüz ve yaratıcılığınız, aldatma veya başkalarının zararına kullanılabilir.   ")
        col_devil.write(
            " 👹  Empati yeteneğiniz, başkalarını manipüle etme ve kendi çıkarlarınız doğrultusunda kullanma potansiyeline sahiptir.   ")
        col_devil.write(
            " 👹 Değişken doğanız, etrafınızdakileri etkilemek ve kontrol altına almak için kullanılabilir. ")
        col_devil.write(
            " 👹  Pozitif enerjiniz, başkalarını etkileme konusundaki becerinizle birleşerek, karanlık bir liderlik potansiyeli oluşturabilir.   ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Siz, Hayal gücü dolu bir vizyoner olarak, yaratıcı projelerde bulunma ve çevrenizdeki insanları olumlu bir şekilde etkileme yeteneğiniz var.  ")
        col_angel.write(
            " 😇 Empati ve anlayışınız, duygusal ihtiyaçları anlama konusundaki doğal yeteneklerinizle birleşiyor.  ")
        col_angel.write(
            " 😇 Değişime hızla adapte olabilme esnekliğiniz, farklı sosyal ortamlarda başarıya ulaşmanıza katkıda bulunuyor. ")
        col_angel.write(
            " 😇 Pozitif enerjiniz, zorluklarla karşılaştığınızda bile etrafınızdakilere ilham kaynağı oluyor. Grup içinde uyum sağlama ve olumlu bir katkı sunma isteğiniz, sizi melek gibi bir karakter yapar. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦋TABİİ Kİ KELEBEK!🦋")
        animal1.image("img/animals/animal5.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write("Ruh hayvanınız, renkli ve özgür bir kelebek. Her bir kanadında değişiklik ve dönüşüm sembolleri taşıyarak, sürekli bir büyüme ve evrim içindesiniz. Kelebek gibi, enerjinizi çevrenizde yayarak olumlu bir etki bırakma yeteneğiniz var. Hayal gücünüz ve yaratıcılığınız, çiçekleri ziyaret eden kelebek gibi, etrafınıza güzellik ve ilham katıyor. Siz, özgürlük ve değişim arayışınızda, kelebek gibi renkli ve dikkat çekici bir varlıksınız")
    elif personality_type == "ISFP":
        tab_model.write("## 📌 AÇIK BİR ZİHİN, PARLAK BİR ZEKA  ")
        tab_model.write(
            "Siz, hayata esnek ve maceraperest bir yaklaşım getiriyorsunuz. Günlük rutinlere bağlı kalmak yerine her anın tadını çıkarıyor ve beklenmedik durumlara karşı açık bir zihinle yaklaşıyorsunuz. Bu esneklik sizi hoşgörülü ve çeşitli dünya görüşlerine karşı açık fikirli kılıyor. Her günü bir macera olarak görüyorsunuz ve değerli anılarınızı spontane anlar, geziler ve beklenmedik maceralarla dolu kılıyorsunuz. Ayrıca, farklı yaşam tarzlarına ve insanlara olan toleransınız, değişen bakış açılarına açık olmanızla öne çıkıyor. ")
        tab_model.write("## 📌 SIRADIŞI TUTKULARIN PEŞİNDE ")
        tab_model.write(
            "Siz, her ne kadar geleneksel anlamda olmasa da, gerçek bir sanatçısınız. Sizin için hayat, kendinizi ifade etmek için bir tuvaldir. Ne giydiğinizden boş zamanlarınızı nasıl geçirdiğinize kadar, benzersiz bireyler olarak kim olduğunuzu canlı bir şekilde yansıtacak şekilde hareket edersiniz. Siz kesinlikle kendinizde özgüsünüz. Merakla hareket eden ve yeni şeyler denemeye istekli olan karakteriniz, genellikle büyüleyici bir tutku ve ilgi alanına sahiptir.  ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  İçsel değerler ve duygular sizin için önemli, ve bu özellikleri kötü niyetli bir şekilde kullanabilirsiniz. Duygusal Manipülasyon başkalarını etkileme konusundaki en tehlikeli silahınız.   ")
        col_devil.write(
            " 👹  Stres altında ve duygusal zorluklarda, intikamcı bir kişiliğe bürünebilirsiniz. Özellikle biri sizin değerlerinize zarar veriyorsa, intikamcılığınız daha belirin hale gelebilir.  ")
        col_devil.write(
            " 👹 Kararsızlıkları ve çatışmaları sevmeyişiniz, sizi sorumluluk almaktan kaçan bir karaktere dönüştürebilir. ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Sizin derin dünyanız muhteşem bir yaratıcılık barındırıyor.  ")
        col_angel.write(
            " 😇 Sanatsal ifadelerle ilgileniyor, duygusal olarak derin bağlar kurup ve çevrenizdeki insanların ihtiyaçlarına duyarlı oluyorsunuz.  ")
        col_angel.write(
            " 😇 Her ortama karşı esnek ve uyumlusunuz. Değişen durumlara karşı hızlıca adapte olabilir, çevredeki insanlarla uyum sağlayabilirsiniz. ")
        col_angel.write(
            " 😇 Güçlü empati yeteneğiniz ile insanları daha iyi anlayıp onlara yardım edebiliyorsunuz. ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦌TABİİ Kİ Geyik!🦌")
        animal1.image("img/animals/animal9.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Sizin kişiliğiniz bir Geyik gibi zarafeti, doğayı ve duyarlılığı temsil ediyor. Sakin ve içsel bir güzelliğin sembolü olan geyik, sizin özgün ve derin düşünen doğanızı yansıtır. Adeta bir geyik gibi, çevrenize huzur ve dinginlik getirirsiniz. ")
    elif personality_type == "ENTP":
        tab_model.write("## 📌 KURALLAR ÇİĞNENMEK İÇİN VAR! ")
        tab_model.write(
            "Siz, isyankar yanınızla tanınırsınız. Hiçbir inanç, fikir veya kural sizin için sorgulanamaz veya kutsal değildir; aksine, her şeyin test edilebilir olduğuna inanırsınız. Karşıt görüşü savunarak dahi kendi inançlarınıza meydan okursunuz ve düşünce tarzlarını sorgulamaktan keyif alırsınız. Sosyal normlara meydan okuma ve farklı bakış açılarını ortaya çıkarma konusunda isteklisiniz. Beyin fırtınası yapmaktan hoşlanırsınız ancak fikirleri uygulamaktan kaçınabilir ve öncelik belirleme konusunda zorlanabilirsiniz. ")
        tab_model.write("## 📌 ANA MUHALEFET  ")
        tab_model.write(
            "Sizin tartışma kabiliyetleriniz olağanüstü. Mantık ve rasyonalitenin ön planda olduğu bir dünyada, açıkça düşüncelerinizi ifade etmekten çekinmezsiniz. Ancak bu, her zaman hayatınızı kolaylaştırmaz. Patronlarınızı sorgulamak veya çekişmeli eğlencelerle ilişkileri zora sokmak gibi durumlarla karşılaşabilirsiniz. Güçlü görüşleriniz ve mizah anlayışınızla saygı görebilirsiniz, ancak duyarlılık konusunda gelişme sağlamazsanız, daha derin ilişkiler kurmak veya profesyonel hedeflerinize ulaşmak konusunda zorluklar yaşayabilirsiniz.")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Muhteşem karşı argüman sunma yetenekleriniz, insanları kolayca manipüle edebilir. ")
        col_devil.write(
            " 👹  Duygusal olarak bağlı olduğunuz insanları istediğinizde hayatınızdan çok kolay çıkartabiliyorsunuz.")
        col_devil.write(
            " 👹 Kolay sıkılan yapınız, insanların fikirlerini umursamayan bir karaktere dönüştürebilir. ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Yenilikçi düşünen ve doğuştan bir alternatif üreticisisiniz.  ")
        col_angel.write(
            " 😇Düşüncelerinizi tutkulu ve canlı bir şekilde aktarabilip insanları etkileyebiliyorsunuz. ")
        col_angel.write(
            " 😇 Emrederek değil, yaparak ilham olan bir lider yapınız var.  ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🦊TABİİ Kİ Tilki!🦊")
        animal1.image("img/animals/animal16.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Siz hayvanlar aleminin muhalefetisiniz, yani Tilki. Tilki kurnazlığı ve çevredeki yaşanan şeylere farkında olmayı temsil eder. Siz, kurnaz ve zeki bir karakter olarak insanları zekice analiz etmeyi seviyorsunuz. Konu hayvanlar aleminin kralına karşı çıkmak olduğunda, aslana muhalefet eden tilkidir.")
    elif personality_type == "ISFJ":
        tab_model.write("## 📌 SADAKAT KUTSALDIR  ")
        tab_model.write(
            "Sizin en belirgin özelliğiniz sadakatinizdir. Siz, bir arkadaşınızın veya bir aile üyenizin zor zamanında her şeyi bırakıp yardım eden o kötü gün dostusunuz. Sadakatiniz alçakgönüllü ve gösterişsizdir, çalışkanlığınız ile çevrenizdekilere karşı derin bir sorumluluk hissi taşımaktasınız.  ")
        tab_model.write("## 📌 LİMİT GÖKYÜZÜ ")
        tab_model.write(
            "Konu standartlarınızı belirlemek olduğunda sizin için limit gökyüzüdür, yani limitiniz yok. Yeterince iyi olmak sizin için yeterli değil. Siz en iyisini yapacak, en mükemmeli olacaksınız. Sorumluluklarınız sizin için basit işler değil, tamamlanması gereken kutsal görevlerdir.   ")
        col_devil, col_angel = tab_model.columns(2)
        col_devil.markdown("<h1 style='text-align: left; color: red;'> 👺 EVIL YOU 👺</h1>", unsafe_allow_html=True)
        col_devil.image("img/VK/1.png", width=300)
        col_devil.write(
            " 👹  Herkesin sizin kabul ettiğiniz değerler ve geleneklere inanması gerektiğini düşünüyorsunuz. Yanlış olsa bile. ")
        col_devil.write(
            " 👹  Biri sizin geleneklerinizi ve değerlerinizi ihlal ederse, onları cezalandırmaktan kaçınmazsınız.  ")
        col_devil.write(
            " 👹 İçinizden geldiği için değil, sadece iltifat alabilmek için iyi şeyler yapma eğilimindesiniz. ")
        col_angel.markdown("<h1 style='text-align: left; color: cyan;'>👼🏻 ANGEL YOU 👼🏻</h1>", unsafe_allow_html=True)
        col_angel.image("img/VK/2.png", width=300)
        col_angel.write(
            " 😇 Son derece korumacısınız, sevdiğiniz insanları korumak sizin için en önemli şey.   ")
        col_angel.write(
            " 😇Sakin, dost canlısı, sorumluluk sahibi ve bilinçlisiniz.  ")
        col_angel.write(
            " 😇 Verdiği sözleri karşılamada son derece kararlı ve istikrarlısınız ")
        ## spirit animals
        animal1, animal2 = tab_model.columns(2)
        animal1.write("# Ruh hayvanınız.. ")
        animal1.write("## 🐧TABİİ Kİ Penguen!🐧")
        animal1.image("img/animals/animal1.jpg", width=350)
        animal2.write("## Neden?")
        animal2.write(
            "Sadık, koruyucu ve zeki. Sizin ruh hayvanınız bir penguen. Tıpkı penguenler gibi, sizin karakteriniz de aşırı sadık ve koruyucu. Penguenler gibi, en yakınlarına olan sadakatiniz çok güçlü. Penguenler son derece dikkatli ve odaklıdır. Tıpkı sizin gibi, yaptıkları işleri son derece titiz ve kusursuz şekilde yaparlar.  ")

    # Add more conditions for other personality types

if __name__ == "__main__":
    main()
