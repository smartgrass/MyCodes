Shader "MyShader/anim_vert"{
    Properties{
        _MainTex("Animation Texture",2D) = "white"{}
        _BaseColor("BaseColor",Color) = (1.0,1.0,1.0,1.0)
        _XSpeed("_XSpeed",float) = 1
        _YSpeed("_YSpeed",float) = 1
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
            float _XSpeed;
            float _YSpeed;
            float _Speed;

            vertexOutput vert(appdata_base v){

                vertexOutput o;
				v.vertex.y += sin(_Time.y + v.vertex.x * _XSpeed + v.vertex.z * _YSpeed);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.texcoord;
                return o;
            }

            fixed4 frag(vertexOutput i):SV_TARGET{
                fixed4 c = tex2D(_MainTex,i.uv);
                return c * _BaseColor;
            }
            ENDCG
        }
    }
}