<script setup>
import axios from 'axios'
import { ref } from 'vue'

const axios_instance = axios.create({ baseURL: 'http://localhost:5000' })
const root_prefix = 'http://localhost:5173/@fs'

let placeholder_appearance =
  'The dress the person wears has long sleeves and it is of short length. Its texture is pure color.'
let placeholder_motion =
  'The lady moves to the right.\n' +
  'The person is moving to the center from the right.\n' +
  'She turns right from the front to the side.\n' +
  'She turns right from the side to the back.'

const input_appearance = ref(placeholder_appearance)
const input_motion = ref(placeholder_motion)
const img = ref(null)
const video = ref(null)

function generate_appearance() {
  axios_instance
    .post('/text2performer/generate_appearance', {
      input_appearance: input_appearance.value
    })
    .then((res) => {
      img.value.src = root_prefix + res.data
    })
}

function generate_motion() {
  axios_instance
    .post('/text2performer/generate_motion', {
      input_motion: input_motion.value
    })
    .then((res) => {
      video.value.src = root_prefix + res.data
    })
}

function interpolate() {
  axios_instance.get('/text2performer/interpolate').then((res) => {
    video.value.src = root_prefix + res.data
  })
}
</script>

<template>
  <div class="main">
    <div class="input_box">
      <div>
        <h1>外貌</h1>
        <textarea
          rows="6"
          cols="50"
          :placeholder="placeholder_appearance"
          v-model="input_appearance"
        ></textarea>
      </div>
      <button type="button" @click="generate_appearance">生成</button>
    </div>
    <img src="/exampler.png" ref="img" />

    <div class="input_box">
      <div>
        <h1>动作</h1>
        <textarea
          rows="6"
          cols="50"
          :placeholder="placeholder_motion"
          v-model="input_motion"
        ></textarea>
      </div>
      <button type="button" @click="generate_motion">生成</button>
      <button type="button" @click="interpolate">插帧</button>
    </div>
    <video src="/video.mp4" ref="video" controls autoplay loop muted></video>
  </div>
</template>

<style scoped>
.main {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.input_box {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

textarea {
  font-size: 15px;
  resize: none;
}

button {
  font-size: 20px;
  width: 100px;
  height: 50px;
  align-self: center;
}
</style>
