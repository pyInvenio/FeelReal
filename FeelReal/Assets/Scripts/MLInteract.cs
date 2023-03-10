using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using UnityEngine;
using System.Threading.Tasks;

public class MLInteract : MonoBehaviour
{
    public string emotion;


    void Start()
    {

    }

    public async Task getMood()
    {
        Debug.Log("Get MOOD");

        //Set a timer for 10s
        //await getAPI("http://127.0.0.1:8000/ser");

        string[] moodes = new string[] { "\"CALM\"", "\"ANGRY\"", "\"DISGUST\"", "\"FEARFUL\"", "\"HAPPY\"", "\"NEUTRAL\"", "\"SAD\"", "\"SURPRISED\"" };
        
        //Random random = new Random();
        emotion = moodes[(int)Random.Range(0, moodes.Length)];

        
        playSounds();
    }
    private async Task getAPI(string api)
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(api);
        using (HttpWebResponse response = (HttpWebResponse)await request.GetResponseAsync())
        {
            using (StreamReader reader = new StreamReader(response.GetResponseStream()))
            {
                string jsonResponse = await reader.ReadToEndAsync();
                emotion = jsonResponse;
            }
        }
        return;
    }

    private void playSounds()
    {
        // get the script attached to the heart
        GameObject heart = GameObject.FindWithTag("Heart");
        GameObject bg = GameObject.Find("BackgroundChanger");
        BackgroundChanger bgChanger = bg.GetComponent<BackgroundChanger>();
        ChangeShader heartScript = heart.GetComponent<ChangeShader>();
        AudioSource audio = GameObject.FindWithTag("as").GetComponent<AudioSource>();
        switch (emotion)
        {
            case "\"CALM\"":
                audio.clip = Resources.Load("Calm") as AudioClip;
                heartScript.ChangeColor(new Color(0, 26 / 255f, 1, 1), new Color(0, 1, 209 / 255f, 1));
                bgChanger.SetNewColors(new Color(19/255f, 29/255f, 64 / 255f), new Color(116 / 255f, 142 / 255f, 233 / 255f), 0.28f, 3);
                break;
            case "\"ANGRY\"":
                audio.clip = Resources.Load("Angry") as AudioClip;
                heartScript.ChangeColor(new Color(0, 148 / 255f, 1, 1), new Color(1, 0, 0, 1));
                bgChanger.SetNewColors(new Color(67 / 255f, 36 / 255f, 8 / 255f), new Color(214 / 255f, 117 / 255f, 28 / 255f), 0.28f, 3);
                break;
            case "\"DISGUST\"":
                audio.clip = Resources.Load("Disgust") as AudioClip;
                heartScript.ChangeColor(new Color(0, 1, 102 / 255f, 1), new Color(1, 61 / 255f, 0, 1));
                bgChanger.SetNewColors(new Color(67 / 255f, 36 / 255f, 8 / 255f), new Color(214 / 255f, 117 / 255f, 28 / 255f), 0.28f, 3);
                break;
            case "\"FEARFUL\"":
                audio.clip = Resources.Load("Fearful") as AudioClip;
                heartScript.ChangeColor(new Color(1 / 255f, 49 / 255f, 119 / 255f, 1), new Color(117 / 255f, 9 / 255f, 168 / 255f, 1));
                bgChanger.SetNewColors(new Color(19 / 255f, 29 / 255f, 64 / 255f), new Color(0 / 255f, 13 / 255f, 134 / 255f), 0.28f, 3);
                break;
            case "\"HAPPY\"":
                audio.clip = Resources.Load("Joyful") as AudioClip;
                heartScript.ChangeColor(new Color(1, 0, 0, 1), new Color(1, 184 / 255f, 0, 1));
                bgChanger.SetNewColors(new Color(19 / 255f, 29 / 255f, 64 / 255f), new Color(116 / 255f, 142 / 255f, 233 / 255f), 0.28f, 3);
                break;
            case "\"NEUTRAL\"":
                audio.clip = Resources.Load("Neutral") as AudioClip;
                heartScript.ChangeColor(new Color(1, 1, 1, 1), new Color(1, 1, 1, 1));
                bgChanger.SetNewColors(new Color(19 / 255f, 29 / 255f, 64 / 255f), new Color(0 / 255f, 13 / 255f, 134 / 255f), 0.28f, 3);
                break;
            case "\"SAD\"":
                audio.clip = Resources.Load("Sad") as AudioClip;
                heartScript.ChangeColor(new Color(0, 0, 0, 1), new Color(0, 26 / 255f, 1, 1));
                bgChanger.SetNewColors(new Color(19 / 255f, 29 / 255f, 64 / 255f), new Color(0 / 255f, 13 / 255f, 134 / 255f), 0.28f, 3);
                break;
            case "\"SURPRISED\"":
                audio.clip = Resources.Load("Surprise") as AudioClip;
                heartScript.ChangeColor(new Color(191 / 255f, 226 / 255, 91 / 255f, 1), new Color(1, 0, 0, 1));
                bgChanger.SetNewColors(new Color(67 / 255f, 36 / 255f, 8 / 255f), new Color(214 / 255f, 117 / 255f, 28 / 255f), 0.28f, 3);
                break;
        }
        Debug.Log(emotion);
        // call the function on the script
        heartScript.ChangeAudio(SetSampleRate(audio.clip, 44100));

        GameManager.isPlaying = true;
        GameManager.clipStart = 0;
    }

    public static AudioClip SetSampleRate(AudioClip clip, int frequency)
    {
        if (clip.frequency == frequency) return clip;
        if (clip.channels != 1 && clip.channels != 2) return clip;

        var samples = new float[clip.samples * clip.channels];

        clip.GetData(samples, 0);

        var samplesNewLength = (int)(frequency * clip.length) * clip.channels;
        var clipNew = AudioClip.Create(clip.name + "_" + frequency, samplesNewLength, clip.channels, frequency, false);

        var channelsOriginal = new List<float[]>();
        var channelsNew = new List<float[]>();

        if (clip.channels == 1)
        {
            channelsOriginal.Add(samples);
            channelsNew.Add(new float[(int)(frequency * clip.length)]);
        }
        else
        {
            channelsOriginal.Add(new float[clip.samples]);
            channelsOriginal.Add(new float[clip.samples]);

            channelsNew.Add(new float[(int)(frequency * clip.length)]);
            channelsNew.Add(new float[(int)(frequency * clip.length)]);

            for (var i = 0; i < samples.Length; i++)
            {
                channelsOriginal[i % 2][i / 2] = samples[i];
            }
        }

        for (var c = 0; c < clip.channels; c++)
        {
            var index = 0;
            var sum = 0f;
            var count = 0;
            var channelSamples = channelsOriginal[c];

            for (var i = 0; i < channelSamples.Length; i++)
            {
                var index_ = (int)((float)i / channelSamples.Length * channelsNew[c].Length);

                if (index_ == index)
                {
                    sum += channelSamples[i];
                    count++;
                }
                else
                {
                    channelsNew[c][index] = sum / count;
                    index = index_;
                    sum = channelSamples[i];
                    count = 1;
                }
            }
        }

        float[] samplesNew;

        if (clip.channels == 1)
        {
            samplesNew = channelsNew[0];
        }
        else
        {
            samplesNew = new float[channelsNew[0].Length + channelsNew[1].Length];

            for (var i = 0; i < samplesNew.Length; i++)
            {
                samplesNew[i] = channelsNew[i % 2][i / 2];
            }
        }

        clipNew.SetData(samplesNew, 0);

        return clipNew;
    }

}
