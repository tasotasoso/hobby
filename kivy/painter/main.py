from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle,Line
from random import random
from kivy.uix.button import Button

class MyPaintWidget(Widget):
    """
    キャンバスのクラス

    Notes
    -----
    on_touch_down()はクリックイベント時のアクションをオーバーライド。
    on_touch_move()はドラッグイベント時のアクションをオーバーライド。 
    """

    def on_touch_down(self, touch):
        color = (random(), random(), random())
        with self.canvas: #withでアクションの定義を行う。要素を組み込んでいくイメージ？
            Color(*color)
            d = 30.
            Rectangle(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y),width = 10)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class MyPaintApp(App):
    """
    アプリケーションのクラス

    Notes
    -----
    build：アプリケーションの構成要素インスタンスを生成して組み立てるイメージ
    #---#：この間にbuildで作る構成要素にバインドする機能を書いている。

    """

    def build(self):

        #構成要素をまとめるparentの作成
        parent = Widget()

        #キャンバスの設定
        painter = MyPaintWidget()

        #クリアボタンの設定        
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)

        #parentにキャンバスとクリアボタンを追加する
        parent.add_widget(painter)
        parent.add_widget(clearbtn)

        return parent

    #--------------------------------#
    #ウィジェットにバインドする機能の定義
    def clear_canvas(self, obj):
        self.painter.canvas.clear()


    #--------------------------------#

if __name__ == "__main__":
    MyPaintApp().run()
