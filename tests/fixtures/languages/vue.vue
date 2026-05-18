<template>
  <section class="language-panel">
    <h1>{{ title }}</h1>
    <button type="button" @click="increment">{{ count }}</button>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

const props = defineProps<{ initial: number }>()
const emit = defineEmits<{ changed: [value: number] }>()

const count = ref(props.initial)
const title = computed(() => `Vue count ${count.value}`)

function increment() {
  count.value += 1
  emit('changed', count.value)
}
</script>

<style scoped>
.language-panel {
  display: grid;
  gap: 0.75rem;
}
</style>
