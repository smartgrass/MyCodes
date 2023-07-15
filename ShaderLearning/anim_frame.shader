Shader "MyShader/anim_frame"{
    Properties{
        _MainTex("Animation Texture",2D) = "white"{}
        _BaseColor("BaseColor",Color) = (1.0,1.0,1.0,1.0)
        _XCount("XCount",Int) = 1
        _YCount("YCount",Int) = 1
        _Speed("Speed",Range(1,100)) = 30
    }
    SubShader{
        Tags{"Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent"}
        pass{
            Tags{"LightMode"="Always"}
            ZWrite Off
            Blend SrcAlpha OneMinusSrcAlpha

            CGPROGRAM
            #include "Lighting.cginc"
            #include "UnityCG.cginc"
            #include "AutoLight.cginc"

            #pragma vertex vert
            #pragma fragment frag

            struct vertexOutput{

                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            sampler2D _MainTex;
            fixed4 _BaseColor;
            int _XCount;
            int _YCount;
            float _Speed;

            vertexOutput vert(appdata_base v){

                vertexOutput o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                return o;
            }
			//核心逻辑
            fixed4 frag(vertexOutput i):SV_TARGET{

				//时间乘速度并取整
                float frame = floor(_Time.y * _Speed);

				//列位置 = 列数 (向下取整)
                float ypos = floor(frame / _XCount);
				//行位置 = 序号 - 列数 * 每列个数
                float xpos = frame - ypos * _XCount;

                i.uv.x = (i.uv.x + xpos) / _XCount;
                i.uv.y = 1 - (ypos + 1 - i.uv.y)/_YCount;
                //根据我们刚才计算出来的公式进行uv坐标的偏移和缩放

                fixed4 c = tex2D(_MainTex,i.uv);
                return c * _BaseColor;
            }
            ENDCG
        }
    }
}